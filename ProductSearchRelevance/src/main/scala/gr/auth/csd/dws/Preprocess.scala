package gr.auth.csd.dws

import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.feature.Stemmer

import gr.auth.csd.dws.IOUtils.log



object Preprocess {

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
  def stemming(dataFrame: DataFrame):DataFrame = {
    val df_with_stemmed_title = titleStemming(dataFrame)
    val df_with_stemmed_descriptions = descriptionStemming(df_with_stemmed_title)
    val df_with_stemmed_attributes = attributeStemming(df_with_stemmed_descriptions)

//    searchQuerySpellCorrector()

    df_with_stemmed_attributes
  }


  private def titleStemming(dataFrame: DataFrame):DataFrame = {
    stemming(dataFrame, "product_title", "stemmed_product_title")
  }

  private def descriptionStemming(dataFrame: DataFrame):DataFrame = {
    stemming(dataFrame, "product_description", "stemmed_product_description")
  }

  private def attributeStemming(dataFrame: DataFrame):DataFrame = {
    val df_with_stemmed_name = stemming(dataFrame, "name", "stemmed_name")
    val df_with_stemmed_value = stemming(df_with_stemmed_name, "value", "stemmed_value")

    df_with_stemmed_value
  }

//  private def searchQuerySpellCorrector(dataFrame: DataFrame):DataFrame = {
//
//  }


  private def stemming(dataFrame: DataFrame, inputCol:String, outputCol:String):DataFrame = {
    val stemmed = new Stemmer().setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setLanguage("English")
      .transform(dataFrame)

    log("Stemming '" + inputCol + "' into '" + outputCol + "'")
    stemmed.show(10)

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
