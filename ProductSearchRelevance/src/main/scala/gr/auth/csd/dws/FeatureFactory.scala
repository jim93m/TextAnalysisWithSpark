package gr.auth.csd.dws

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._
import gr.auth.csd.dws.IOUtils.log


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
    val df_with_cosine_distance = FeatureFactory.cosine_distance(spark, df)
    val df_with_tf_idf = FeatureFactory.tf_idf(spark, df_with_cosine_distance)
    val df_with_len_of_query = FeatureFactory.len_of_query(df_with_tf_idf)
    val df_with_lev_dist_prod_desc = FeatureFactory.levenshtein_distance(df_with_len_of_query, col("stemmed_product_description"), col("stemmed_search_term"), "levenshtein_dist_description")
    val df_with_lev_dist_prod_title = FeatureFactory.levenshtein_distance(df_with_lev_dist_prod_desc, col("stemmed_product_title"), col("stemmed_search_term"), "levenshtein_dist_title")
//    val df_with_lev_dist_attr = FeatureFactory.levenshtein_distance(df_with_lev_dist_prod_title, col("value"), col("stemmed_search_term"), "levenshtein_dist_attr")


    df_with_lev_dist_prod_title
  }

  private def cosine_distance(spark:SparkSession, df:DataFrame):DataFrame = {

    df
  }

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
    dataFrame.withColumn("len_of_query", searchTermLength(col("search_term")))
  }
  private val searchTermLength = udf((searchTerm: String) => searchTerm.length())

  /**
    * # New Feature: Length of the product brand string (if it occurs)
    * df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(str(x).split())).astype(np.int64)
    */

  /**
    *
    * # New Feature: Length of attribute string for each product instance
    * df_all['len_of_attribute'] = df_all['attribute'].map(lambda x:len(str(x).split())).astype(np.int64)
    */

  /**
    *
    * # Internal Feature: Product Info. This collects the related search term, title, description, and product attributes.
    * # Note that this is obviously not used in the model given that it is a string.
    * # Rather, this is used in subsequent calculations.
    * df_all['product_info'] = df_all['search_term']+'\t'+df_all['product_title']+'\t'+df_all['product_description']+'\t'+df_all['attribute']
    */

  /**
    *
    * # New Feature: Number of words which occur in both the query and in the product's title
    * df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(str(x).split('\t')[0],str(x).split('\t')[1]))
    */

  /**
    *
    * # New Feature: Number of words which occur in both the query and in the product's description.
    * # Again, product_info = product_title + product_description + attribute
    * df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(str(x).split('\t')[0],str(x).split('\t')[2]))
    */

  /**
    *
    * # New Feature: Number of words which occur in both the query and in the product's attributes.
    * df_all['word_in_attributes'] = df_all['product_info'].map(lambda x:str_common_word(str(x).split('\t')[0],str(x).split('\t')[3]))
    */

  /**
    *
    * # New Feature: Number of words which occur in both the query and in the product's brand (if defined)
    * df_all['attr'] = str(df_all['search_term'])+"\t"+str(df_all['brand'])
    * df_all['brand_in_search'] = df_all['attr'].map(lambda x:str_common_word(str(x).split('\t')[0],str(x).split('\t')[1]))
    */

  /**
    *
    * # New Feature: Ratio of terms which are common between the brand and query, and the overall length of the brand string
    * df_all['ratio_brand'] = df_all['brand_in_search']/df_all['len_of_brand']
    */

  /**
    *
    * # New Feature: Ratio of terms which are common in the title and the overall length of the query string
    * df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
    */

  /**
    *
    * # New Feature: Ratio of terms which are common in the description and the overall length of the query string
    * df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
    */

  /**
    *
    * # New Feature: Ratio of terms which are common in the attributes and the overall length of the query string
    * df_all['ratio_attributes'] = df_all['word_in_attributes']/df_all['len_of_query']
    */

  /**
    *
    * # New Feature: The last word which matched between the title and the query
    * df_all['last_word_title_match'] = df_all['product_info'].map(lambda x:str_common_word(str(x).split('\t')[0].split(" ")[-1],str(x).split('\t')[1]))
    */

  /**
    *
    * # New Feature: The last word which matched between the description and the query
    * df_all['last_word_description_match'] = df_all['product_info'].map(lambda x:str_common_word(str(x).split('\t')[0].split(" ")[-1],str(x).split('\t')[2]))
    */

  /**
    *
    * # New Feature: The first word which matched between the title and the query
    * df_all['first_word_title_match'] = df_all['product_info'].map(lambda x:str_common_word(str(x).split('\t')[0].split(" ")[0],str(x).split('\t')[1]))
    */

  /**
    *
    * # New Feature: The first word which matched between the description and the query
    * df_all['first_word_description_match'] = df_all['product_info'].map(lambda x:str_common_word(str(x).split('\t')[0].split(" ")[0],x.split('\t')[2]))
    */

}
