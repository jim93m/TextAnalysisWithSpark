package gr.auth.csd.dws

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.{DataFrame, SparkSession}

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


    val final_df = df_with_tf_idf
    final_df
  }

  private def cosine_distance(spark:SparkSession, df:DataFrame):DataFrame = {

    df
  }

  private def tf_idf(spark:SparkSession, df:DataFrame):DataFrame = {
    val tokenizer = new Tokenizer().setInputCol("product_description").setOutputCol("product_description_words")
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

}
