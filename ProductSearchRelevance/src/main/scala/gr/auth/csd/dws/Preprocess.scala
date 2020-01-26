package gr.auth.csd.dws

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.functions._
import gr.auth.csd.dws.IOUtils.log
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
import org.apache.spark.rdd.RDD



object Preprocess {

  def run(spark: SparkSession, dataFrame: DataFrame, logging: Boolean):DataFrame = {
    val lowerDF = toLowerCase(spark, dataFrame)
//    val removedStopwordsDF = removeStopwords(spark, lowerDF)
//    searchQuerySpellCorrector()
    val stemmedDF = stemming(lowerDF, logging=logging)

    stemmedDF
  }

  /**
    * @param spark the spark session
    * @param dataFrame the dataframe to preprocess
    *
    * >> Product Title Stemming
    * >> Product Description Stemming
    * >> Product Attributes Stemming
    *
    * >> Search Term Spell Check
    *
    * @return pre-processed (stemmed with Snowball Stemmer) dataframe
    */
  private def stemming(dataFrame: DataFrame, logging: Boolean):DataFrame = {
    val df_with_stemmed_title = titleStemming(dataFrame, logging=logging)
    val df_with_stemmed_descriptions = descriptionStemming(df_with_stemmed_title, logging=logging)
    val df_with_stemmed_attributes = attributeStemming(df_with_stemmed_descriptions, logging=logging)
    val df_with_stemmed_search_terms = searchTermStemming(df_with_stemmed_attributes, logging=logging)

    df_with_stemmed_search_terms
  }


  private def titleStemming(dataFrame: DataFrame, logging: Boolean):DataFrame = {
    stemming(dataFrame, "product_title", "stemmed_product_title", logging=logging)
  }

  private def descriptionStemming(dataFrame: DataFrame, logging: Boolean):DataFrame = {
    stemming(dataFrame, "product_description", "stemmed_product_description", logging=logging)
  }

  private def attributeStemming(dataFrame: DataFrame, logging: Boolean):DataFrame = {
    stemming(dataFrame, "attributes", "stemmed_attributes", logging=logging)
  }

  private def searchTermStemming(dataFrame: DataFrame, logging: Boolean):DataFrame = {
    stemming(dataFrame, "search_term", "stemmed_search_term", logging=logging)
  }

//  private def searchQuerySpellCorrector(dataFrame: DataFrame):DataFrame = {
//
//  }

  private def toLowerCase(spark: SparkSession, dataFrame: DataFrame):DataFrame = {
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
    log("Converting text-columns to lowercase...")
    dataFrame.select($"product_uid",
                      $"id",
                      lower($"product_title").alias("product_title"),
                      lower($"search_term").alias("search_term"),
                      $"relevance",
                      lower($"product_description").alias("product_description"),
                      lower($"attributes").alias("attributes"))
  }

  def dropUnneededColumns(spark: SparkSession, dataFrame: DataFrame):DataFrame = {
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._

    log("Dropping unneeded colmns (product_title, product_description, attributes, search_term)")
    dataFrame.drop($"product_title")
             .drop($"product_description")
             .drop($"attributes")
             .drop($"search_term")
  }

  def renameColumns(dataFrame: DataFrame):DataFrame = {
    log("Renaming unneeded columns: ")
    log(">> stemmed_product_title        - to -   title")
    log(">> stemmed_product_description  - to -   description")
    log(">> stemmed_attributes           - to -   attributes")
    log(">> stemmed_search_term          - to -   search_term")

    dataFrame.withColumnRenamed("stemmed_product_title", "title")
             .withColumnRenamed("stemmed_product_description", "description")
             .withColumnRenamed("stemmed_attributes", "attributes")
             .withColumnRenamed("stemmed_search_term", "search_term")
  }

  private def stemming(dataFrame: DataFrame, inputCol:String, outputCol:String, logging: Boolean):DataFrame = {
    val stemmed = new Stemmer().setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setLanguage("English")
      .transform(dataFrame)

    log("Stemming '" + inputCol + "' into '" + outputCol + "'")
    if (logging) {
      stemmed.show(10, truncate = 80)
    }

    stemmed
  }

  //  private def removeStopwords(spark: SparkSession, dataFrame: DataFrame):DataFrame = {
  //    val stopWordsInput = spark.sparkContext.textFile("resources/stopwords.csv")
  //    val stopWords = stopWordsInput.flatMap(x => x.split("\\W+")).collect()
  //
  //    import spark.implicits._ // For implicit conversions like converting RDDs to DataFrames
  //
  //    val tokenizedTitleDf = new RegexTokenizer().setInputCol("product_title")
  //                                        .setOutputCol("product_title_tokens")
  //                                        .setPattern("[\\W_]+").transform(dataFrame)
  //
  //    val tokenizedDescriptionDf = new RegexTokenizer().setInputCol("product_description")
  //                                        .setOutputCol("product_description_tokens")
  //                                        .setPattern("[\\W_]+").transform(tokenizedTitleDf)
  //
  //    val tokenizedTermDf = new RegexTokenizer().setInputCol("search_term")
  //                                        .setOutputCol("search_term_tokens")
  //                                        .setPattern("[\\W_]+").transform(tokenizedDescriptionDf)
  //
  //    val tokenizedAttrDf = new RegexTokenizer().setInputCol("attributes")
  //                                        .setOutputCol("attributes_tokens")
  //                                        .setPattern("[\\W_]+").transform(tokenizedTermDf)
  //
  //    val tokenizedWordsDf = tokenizedAttrDf
  //
  //
  //    val rows: RDD[Row] = tokenizedWordsDf.rdd
  //    val changedRowsRDD = rows.map(x => (x(0).toString,//product_uid
  //                                        x(1).toString,//id
  //                                        x(2).toString,//product_title
  //                                        x(3).toString,//search_term
  //                                        x(4).toString,//relevance
  //                                        x(5).toString,//product_description
  //                                        x(6).toString,//attributes
  //                                        stringCleaning(x(7).toString),//product_title_tokens
  //                                        stringCleaning(x(8).toString),//product_description_tokens
  //                                        stringCleaning(x(9).toString),//search_term_tokens
  //                                        stringCleaning(x(10).toString)//attributes_tokens
  //    ))
  //
  //    val cleanWordsDf = spark.createDataFrame(changedRowsRDD).toDF("product_uid", "id", "product_title", "search_term","relevance","product_description","attributes","product_title_tokens","product_description_tokens","search_term_tokens","attributes_tokens")
  //    val tokenizerForCleanWordsDf = new RegexTokenizer().setInputCol("rProductTitleWords").setOutputCol("rChangedProductTitleWords").setPattern("[\\W_]+")
  //    val tokenizedChangedSpecialWordsDf = tokenizerForCleanWordsDf.transform(cleanWordsDf)
  //
  //    /** We now remove the stopwords from tokenizedChangedSpecialWordsDf, and save the final filtered changed words to rFilteredChangedWords column*/
  //    val stopWordRemover = new StopWordsRemover()
  //      .setStopWords(stopWords) // This parameter is optional
  //      .setInputCol("rChangedProductTitleWords")
  //      .setOutputCol("rFilteredProductTitleWords")
  //
  //    val filteredWordsDf = stopWordRemover.transform(tokenizedChangedSpecialWordsDf)
  //
  //
  //    val finalFilteredWordsDf = filteredWordsDf.select($"rId", $"rProductUID", $"rFilteredProductTitleWords",$"rRelevance").withColumnRenamed("rFilteredProductTitleWords","rFilteredWords")
  //
  //    return finalFilteredWordsDf
  //
  //    ("product_uid",
  //      "id",
  //      "product_title",
  //      "search_term",
  //      "relevance",
  //      "product_description",
  //      "attributes")
  //
  //    //      lower($"product_title").alias("product_title"),
  ////      lower($"search_term").alias("search_term"),
  ////      lower($"product_description").alias("product_description"),
  ////      lower($"attributes").alias("attributes"))
  //
  //
  //
  //  }

//  /**
//    * String cleaning
//    *  - Replace unneeded characters/strings with spaces
//    *  - Collect dimensional indication terms (i.e. "num x num") under a common identifier
//    *  - Collect unit instances under common identifiers to ensure TF-IDF accuracy
//    *  - Convert string representations of numbers to numeric ones
//    */
//  def stringCleaning(str2clean: String): String = {
//    str2clean.replaceAll("\\bSome.\\b"," ")
//      .replaceAll("°","degrees")
//      .replaceFirst("WrappedArray","")
//      .replaceAll("""[\p{Punct}&&[^.]]""","")
//      .replaceAll("("," ")
//      .replaceAll("  "," ")
//      .replaceAll("$"," ")
//      .replaceAll("?"," ")
//      .replaceAll("-"," ")
//      .replaceAll("//","/")
//      .replaceAll("..",".")
//      .replaceAll(" / "," ")
//      .replaceAll(" \\ "," ")
//      .replaceAll("."," . ")
//      .replaceAll("*"," xbi ")
//      .replaceAll(" by "," xbi ")
//      .replaceAll("([0-9]+)( *)(inches|inch|in|')", "in. ")
//      .replaceAll("([0-9]+)( *)(foot|feet|ft|'')", "ft. ")
//      .replaceAll("([0-9]+)( *)(pounds|pound|lbs|lb)", "lb. ")
//      .replaceAll("([0-9]+)( *)(square|sq) (feet|foot|ft)", "sq.ft. ")
//      .replaceAll("([0-9]+)( *)(cubic|cu) (feet|foot|ft)", "cu.ft. ")
//      .replaceAll("([0-9]+)( *)(gallons|gallon|gal)", "gal. ")
//      .replaceAll("([0-9]+)( *)(ounces|ounce|oz)", "oz. ")
//      .replaceAll("([0-9]+)( *)(centimeters|cm)", "cm. ")
//      .replaceAll("([0-9]+)( *)(milimeters|mm)", "mm. ")
//      .replaceAll("zero", " 0 ")
//      .replaceAll("one", " 1 ")
//      .replaceAll("two", " 2 ")
//      .replaceAll("three", " 3 ")
//      .replaceAll("four", " 4 ")
//      .replaceAll("five", " 5 ")
//      .replaceAll("six", " 6 ")
//      .replaceAll("seven", " 7 ")
//      .replaceAll("eight",  " 8 ")
//      .replaceAll("nine", " 9 ")
//  }


}
