package gr.auth.csd.dws

import org.apache.spark.ml.feature._
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._
import gr.auth.csd.dws.IOUtils.log
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.sql.types.{StringType, StructField, StructType}


object FeatureFactory {

  /**
    * Features Generation
    *     i. Distances / String similarities
    *        >> TF-IDF distance
    *        >> Levenshtein distance
    *        >> Cosine distance
    *        >> Jaccard distance
    *
    * between:
    *   >> search term and product title
    *   >> search term and product description
    *   >> search term and transposed attributes
    *
    *
    *    ii. Other features
    *        >> Count of each query (frequency) in train/test dataset
    *        >> Count number of bullets in attributes (how may attributes)
    *        >> Count letters histograms for both search query and product title
    *        >> Length of search query, product title, attributes
    *        >> Common words ratio between search term -vs- title, description and attributes
    *
    */
  def generateFeatures(spark: SparkSession, df:DataFrame):DataFrame = {
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._

    log("Calculating TF-IDF (product title, product description, search term)...")
    val df_with_title_tf_idf = FeatureFactory.title_tf_idf(spark, df)
    val df_with_desc_tf_idf = FeatureFactory.description_tf_idf(spark, df_with_title_tf_idf)
    val df_with_term_tf_idf = FeatureFactory.search_term_tf_idf(spark, df_with_desc_tf_idf)

    log("Calculating Similarities (Cosine)...")
    val df_with_cosine_distance = FeatureFactory.cosine_distance(spark, df_with_term_tf_idf)
//    val cosine_distance_DF = FeatureFactory.cosine_distance(spark, df_with_term_tf_idf)
    val jaccard_similarity_DF = FeatureFactory.jaccard_similarity(spark, df_with_term_tf_idf)

    log("Calculating Levenshtein Distances...")
    val df_with_lev_dist_prod_desc = FeatureFactory.levenshtein_distance(df_with_cosine_distance, col("stemmed_product_description"), col("stemmed_search_term"), "levenshtein_dist_description")
    val df_with_lev_dist_prod_title = FeatureFactory.levenshtein_distance(df_with_lev_dist_prod_desc, col("stemmed_product_title"), col("stemmed_search_term"), "levenshtein_dist_title")
    val df_with_lev_dist_attr = FeatureFactory.levenshtein_distance(df_with_lev_dist_prod_title, col("stemmed_attributes"), col("stemmed_search_term"), "levenshtein_dist_attr")

    log("Calculating length of each search term, length of attributes, number of attributes per product...")
    val df_with_len_of_query = FeatureFactory.len_of_query(df_with_lev_dist_attr)
    val df_with_len_of_attr= FeatureFactory.len_of_attributes(df_with_len_of_query)
    val df_with_num_of_attr= FeatureFactory.num_of_attributes(df_with_len_of_attr)

    log("Calculating words in common...")
    val df_with_commonWords_term_title = FeatureFactory.common_words(spark, df_with_num_of_attr, $"stemmed_search_term", $"stemmed_product_title", "commonWords_term_title")
    val df_with_commonWords_term_description = FeatureFactory.common_words(spark, df_with_commonWords_term_title, $"stemmed_search_term", $"stemmed_product_description", "commonWords_term_description")
    val df_with_commonWords_term_attr = FeatureFactory.common_words(spark, df_with_commonWords_term_description, $"stemmed_search_term", $"stemmed_attributes", "commonWords_term_attr")

    log("Calculating ratio of words in common...")
    val df_with_ratio_of_commonWords_term_title = FeatureFactory.ratio_of_common_words(df_with_commonWords_term_attr, $"commonWords_term_title", $"len_of_query", "ratio_title")
    val df_with_ratio_of_commonWords_term_desc = FeatureFactory.ratio_of_common_words(df_with_ratio_of_commonWords_term_title, $"commonWords_term_description", $"len_of_query", "ratio_description")
    val df_with_ratio_of_commonWords_term_attr = FeatureFactory.ratio_of_common_words(df_with_ratio_of_commonWords_term_desc, $"commonWords_term_attr", $"len_of_query", "ratio_attr")

    df_with_ratio_of_commonWords_term_attr
  }

  private def cosine_distance(spark:SparkSession, dataFrame:DataFrame):DataFrame = {
    val normalized_product_title_idf_DF = new Normalizer()
      .setInputCol("product_title_idf")
      .setOutputCol("normalized_product_title_idf")
      .transform(dataFrame)


    val normalized_search_term_idf_DF = new Normalizer()
      .setInputCol("search_term_idf")
      .setOutputCol("normalized_search_term_idf")
      .transform(normalized_product_title_idf_DF)


    val featuresAndSearchFeaturesDF = normalized_search_term_idf_DF


    /**Create a vector using Vector Assembler from the two normalized columns and name it Similarity*/
    val output = new VectorAssembler()
      .setInputCols(Array("normalized_product_title_idf", "normalized_search_term_idf"))
      .setOutputCol("title_term_similarity")
      .transform(featuresAndSearchFeaturesDF).select("title_term_similarity").rdd

    /**In order to calculate the Similarity between two columns we need to create the Matrix
      * To do that we get the first field of output rdd which contains the Similarity
      * Afterwards create a RowMatrix with the recently created vector
      * We can compute similarities either with a Brute force(simsPerfect) approach or
      * with an approximation(simsEstimate)
      *
      */
    val items_mllib_vector = output.map(_.getAs[org.apache.spark.ml.linalg.Vector](0)).map(org.apache.spark.mllib.linalg.Vectors.fromML)
    val mat = new RowMatrix(items_mllib_vector)
    /*Brute force approach*/
    //val simsPerfect = mat.columnSimilarities()

    /*With approximation*/
    val simsEstimate = mat.columnSimilarities(0.1) //using DISUM


    /**We now prepare the data to create the new dataframe so that we can use it later on the ML Models*/
    val transformedRDD = simsEstimate.entries.map{case MatrixEntry(row: Long, col:Long, sim:Double) => Array(sim).mkString(",")}
    //Transform rdd[String] to rdd[Row]
    val rdd2 = transformedRDD.map(a => Row(a))

    // to DF
    /**By creating monotonically increasing ids and doing orderby  we ensure that the two dataframes will join properly*/
    val dfschema = StructType(Array(StructField("title_term_similarity",StringType)))
    val rddToDF = spark.createDataFrame(rdd2,dfschema).select("title_term_similarity").withColumn("rowID2",monotonically_increasing_id())

    val relevanceDF = featuresAndSearchFeaturesDF//.select("product_uid","relevance")
      .orderBy("product_uid")
      .withColumn("rowID1",monotonically_increasing_id())

    val cosineDF = relevanceDF.join(rddToDF,relevanceDF("rowID1")===rddToDF("rowID2"),"left")

    cosineDF
  }

  private def jaccard_similarity(spark: SparkSession, dataFrame: DataFrame):DataFrame = {
    val joinedDF = dataFrame.orderBy("id")
    val lsh = new MinHashLSH().setInputCol("search_term_tf").setOutputCol("LSH").setNumHashTables(3)

    val pipe = new Pipeline().setStages(Array(lsh))
    val pipeModel = pipe.fit(joinedDF)

    val transformedDF = pipeModel.transform(joinedDF)
    val transformer = pipeModel.stages

    /*MinHashModel*/
    val tempMinHashModel = transformer.last.asInstanceOf[MinHashLSHModel]
    val threshold = 1.5

    /*Just a udf for converting string to double*/
    val udf_toDouble = udf( (s: String) => s.toDouble )


    /*Perform the Similarity with self-join*/
    /*Find the distance of pairs which is lower than the given threshold*/

    val preSimilarityDF = tempMinHashModel.approxSimilarityJoin(transformedDF,transformedDF,threshold)
      .select(udf_toDouble(col("datasetA.relevance")).alias("relev"),
        col("distCol"))

    /*Make a vector of the distCol and name it Similarity. It will be needed when using the df for the ML Models*/
    val vectorAssem = new VectorAssembler()
      .setInputCols(Array("distCol"))
      .setOutputCol("Similarity")


    val jaccardSimilarityDF = vectorAssem.transform(preSimilarityDF).select("Similarity", "relev").withColumnRenamed("Similarity","jaccard_similarity")
//    val with_jacSim = joinedDF.join(jaccardSimilarityDF, usingColumns = Seq("product_uid"), joinType = "left")

    jaccardSimilarityDF
  }

  private def title_tf_idf(spark:SparkSession, df:DataFrame):DataFrame = {
    val tokenizedDF = tokenize(spark, df, "stemmed_product_title", "product_title_words")
    val tfDF = tf(spark, tokenizedDF, "product_title_words", "product_title_tf")
    val idfDF = idf(spark, tfDF, "product_title_tf", "product_title_idf")

    idfDF
  }

  private def description_tf_idf(spark:SparkSession, df:DataFrame):DataFrame = {
    val tokenizedDF = tokenize(spark, df, "stemmed_product_description", "product_description_words")
    val tfDF = tf(spark, tokenizedDF, "product_description_words", "product_description_tf")
    val idfDF = idf(spark, tfDF, "product_description_tf", "product_description_idf")

    idfDF
  }

  private def search_term_tf_idf(spark:SparkSession, df:DataFrame):DataFrame = {
    val tokenizedDF = tokenize(spark, df, "stemmed_search_term", "search_term_words")
    val tfDF = tf(spark, tokenizedDF, "search_term_words", "search_term_tf")
    val idfDF = idf(spark, tfDF, "search_term_tf", "search_term_idf")

    idfDF
  }

  private def tokenize(sparkSession: SparkSession, dataFrame: DataFrame, inputCol: String, outputCol: String):DataFrame = {
    val tokenizer = new Tokenizer().setInputCol(inputCol).setOutputCol(outputCol)
    tokenizer.transform(dataFrame)
  }

  private def tf(sparkSession: SparkSession, dataFrame: DataFrame, inputCol: String, outputCol: String): DataFrame = {
    val hashingTF = new HashingTF().setInputCol(inputCol).setOutputCol(outputCol).setNumFeatures(20000)
    hashingTF.transform(dataFrame)
  }

  private def idf(sparkSession: SparkSession, dataFrame: DataFrame, inputCol: String, outputCol: String):DataFrame = {
    val idf = new IDF().setInputCol(inputCol).setOutputCol(outputCol)
    val idfModel = idf.fit(dataFrame)
    idfModel.transform(dataFrame)
  }

  private def levenshtein_distance(dataFrame: DataFrame, left:Column, right:Column, newColumnName: String):DataFrame = {
    dataFrame.withColumn(newColumnName, levenshtein(left, right))
  }

  /**
    * 'search_term' length
    */
  private def len_of_query(dataFrame: DataFrame):DataFrame = {
    val searchTermLength = udf((searchTerm: String) => searchTerm.length())
    dataFrame.withColumn("len_of_query", searchTermLength(col("stemmed_search_term")))
  }

  /**
    * 'attributes' length
    */
  private def len_of_attributes(dataFrame: DataFrame):DataFrame = {
    val attributesLength = udf((attributes: String) =>
      if (attributes == null || attributes.isEmpty) {
        0
      } else {
        attributes.length()
      }
    )
    dataFrame.withColumn("len_of_attr", attributesLength(col("stemmed_attributes")))
  }

  /**
    * number of 'attributes'
    */
  private def num_of_attributes(dataFrame: DataFrame):DataFrame = {
    val numOfAttributes = udf((attributes: String) =>
      if (attributes == null || attributes.isEmpty) {
        0
      } else {
        attributes.split("\0").length
      }
    )
    dataFrame.withColumn("num_of_attr", numOfAttributes(col("stemmed_attributes")))
  }

  private def common_words(spark: SparkSession, dataFrame: DataFrame, column1: Column, column2: Column, newColumnName: String):DataFrame = {
    val common_terms = udf((a: String, b: String) =>
      if (a == null || b == null || a.isEmpty || b.isEmpty) {
        0
      } else {
        var tmp1 = a.split(" ")
        var tmp2 = b.split(" ")
        tmp1.intersect(tmp2).length
      }
    )

    dataFrame.withColumn(newColumnName, common_terms(column1, column2))
  }

  private def ratio_of_common_words(dataFrame: DataFrame, commonTermsCol: Column, allTermsCol: Column, newColumnName: String):DataFrame = {
    val ratio_of_common_terms = udf((commonTerms: Float, allTerms: Int) => commonTerms/allTerms)
    dataFrame.withColumn(newColumnName, ratio_of_common_terms(commonTermsCol, allTermsCol))
  }
}
