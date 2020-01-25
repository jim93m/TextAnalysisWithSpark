package gr.auth.csd.dws

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, VectorAssembler}
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._
import gr.auth.csd.dws.IOUtils.log
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.sql.types.{StringType, StructField, StructType}


object FeatureFactory {

  /**
    * Features Generation
    *     i. Distances / String similarities
    *        >> Cosine distance
    *        >> N-gram overlapping distance
    *        >> Longest Common Subsequence
    *        >> Euclidean distance
    *        >> Jaccard distance
    *        >> Jaro distance
    *        >> Smith-Waterman similarity
    *        >> Fuzzy string distances
    *        >> Normalized Compression Distance
    *        >> TF-IDF distance
    *        >> Levenshtein distance
    *
    * between:
    *   >> search term and title using several n-gram segmentations
    *   >> search term and product description
    *   >> search term and transposed attributes
    *   >> search term and bullets, brand, material, color
    *
    *
    *    ii. Other features
    *        >> Count of each query (frequency) in train/test dataset
    *        >> Count of POS tags (Nouns, verbs, etc) using the Stanford Part-Of-Speech Tagger http://nlp.stanford.edu/software/tagger.shtml
    *        >> Count (English) stop words and spelling errors in the query
    *        >> Count number of bullets in attributes (how may attributes)
    *        >> Brand popularity
    *        >> 1st-last word similarities to the title
    *        >> Query-products intra-similarity (i.e. avg and std deviation similarity of the query to all the distinct product it returns) that expresses somehow the ambiguity of the query
    *        >> Count letters histograms for both search query and product title
    *        >> Length of search query, product title, attributes
    *
    */
  def generateFeatures(spark: SparkSession, df:DataFrame):DataFrame = {
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._

    val df_with_cosine_distance = FeatureFactory.cosine_distance(spark, df)
    val df_with_tf_idf = FeatureFactory.tf_idf(spark, df_with_cosine_distance)

    log("Calculating Levenshtein Distances...")
    val df_with_lev_dist_prod_desc = FeatureFactory.levenshtein_distance(df_with_tf_idf, col("stemmed_product_description"), col("stemmed_search_term"), "levenshtein_dist_description")
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


    dataFrame
  }

//  def cosineSimilarity(vectorA: SparseVector, vectorB:SparseVector,normASqrt:Double,normBSqrt:Double) :(Double,Double) = {
//    var dotProduct = 0.0
//    for (i <-  vectorA.indices){
//      dotProduct += vectorA(i) * vectorB(i)
//    }
//    val div = (normASqrt * normBSqrt)
//    if( div == 0 )
//      (dotProduct,0)
//    else
//      (dotProduct,dotProduct / div)
//  }


  private def tf_idf(spark:SparkSession, df:DataFrame):DataFrame = {
    val tokenizer = new Tokenizer().setInputCol("stemmed_product_description").setOutputCol("product_description_words")
    val wordsData = tokenizer.transform(df)

    val hashingTF = new HashingTF()
      .setInputCol("product_description_words").setOutputCol("product_description_tf").setNumFeatures(20)

    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("product_description_tf").setOutputCol("product_description_tfidf")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    val tfidf = rescaledData.select("product_uid", "product_description_tfidf")
    val with_tfidf = df.join(tfidf, usingColumns = Seq("product_uid"), joinType = "left")

    with_tfidf
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
