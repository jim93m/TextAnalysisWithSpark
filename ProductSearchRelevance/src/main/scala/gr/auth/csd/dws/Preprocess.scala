package gr.auth.csd.dws

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.functions._

import gr.auth.csd.dws.IOUtils.log



object Preprocess {

  def run(spark: SparkSession, dataFrame: DataFrame, logging: Boolean):DataFrame = {
    val lowerDF = toLowerCase(spark, dataFrame)
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

//    searchQuerySpellCorrector()

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

  //  private def removeStopwords(dataFrame: DataFrame):DataFrame = {
  //
  //  }


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

  /**
    * String cleaning
    *  - Replace unneeded characters/strings with spaces
    *  - Collect dimensional indication terms (i.e. "num x num") under a common identifier
    *  - Collect unit instances under common identifiers to ensure TF-IDF accuracy
    *  - Convert string representations of numbers to numeric ones
    */
  private def stringCleaning(dataFrame: DataFrame):DataFrame = {
    //  if isinstance(s, str):
    //    s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
    //  s = s.lower()
    //  s = s.replace("  "," ")
    //  s = s.replace(",","") #could be number / segment later
    //    s = s.replace("$"," ")
    //  s = s.replace("?"," ")
    //  s = s.replace("-"," ")
    //  s = s.replace("//","/")
    //  s = s.replace("..",".")
    //  s = s.replace(" / "," ")
    //  s = s.replace(" \\ "," ")
    //  s = s.replace("."," . ")
    //  s = re.sub(r"(^\.|/)", r"", s)
    //  s = re.sub(r"(\.|/)$", r"", s)
    //  s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
    //  s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
    //  s = s.replace(" x "," xbi ")
    //  s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
    //  s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
    //  s = s.replace("*"," xbi ")
    //  s = s.replace(" by "," xbi ")
    //  s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
    //
    //  # Consolidate variations of equivalent unit terms
    //    s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
    //  s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
    //  s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
    //  s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
    //  s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
    //  s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
    //  s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
    //  s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
    //  s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
    //  s = s.replace("°"," degrees ")
    //  s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
    //  s = s.replace(" v "," volts ")
    //  s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
    //  s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
    //  s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
    //  s = s.replace("  "," ")
    //  s = s.replace(" . "," ")
    //
    //  # Handling numeric instances with common identifiers
    //    s = re.sub(r"zero\.?", r"0 ", s)
    //  s = re.sub(r"one\.?", r"1 ", s)
    //  s = re.sub(r"two\.?", r"2 ", s)
    //  s = re.sub(r"three\.?", r"3 ", s)
    //  s = re.sub(r"four\.?", r"4 ", s)
    //  s = re.sub(r"five\.?", r"5 ", s)
    //  s = re.sub(r"six\.?", r"6 ", s)
    //  s = re.sub(r"seven\.?", r"7 ", s)
    //  s = re.sub(r"eight\.?", r"8 ", s)
    //  s = re.sub(r"nine\.?", r"9 ", s)
    //
    //  return s
    //  else:
    //  # Return a "null" string if the parameter supplied is not a string
    //  return "null"



    //Strip the "Bullet" instances (i.e. "Bullet01") which occur in the product attribute names  r"Bullet([0-9]+)"


    dataFrame
  }


}
