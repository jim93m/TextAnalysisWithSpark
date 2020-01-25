package gr.auth.csd.dws

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.sql.functions.{collect_list, concat_ws}
import org.apache.spark.sql.{DataFrame, SparkSession}

object IOUtils {

  def parseAndJoinCSVs(spark: SparkSession, logging: Boolean):DataFrame = {
    /**
      * @train.csv id
      *            product_uid
      *            product_title
      *            search_term
      *            relevance
      * @product_descriptions.csv product_uid
      *                           product_description
      * @attributes.csv product_uid
      *                 name
      *                 value
      */
    val trainDF = csvToDataframe(spark, "train.csv", logging)
    val productDescriptionsDF = csvToDataframe(spark, "product_descriptions.csv", logging)
    val attributesDF_tmp = csvToDataframe(spark, "attributes.csv", logging)

    val attributesDF = flattenAttributes(spark, attributesDF_tmp, logging)

    log("Merging Dataframes...")

    val tmp_df = trainDF.join(productDescriptionsDF, usingColumns = Seq("product_uid"), joinType = "left")
    val merged_df = tmp_df.join(attributesDF, usingColumns = Seq("product_uid"), joinType = "left")

    println(merged_df.show(10, truncate = 80))
    //merged_df.printSchema()
    //df.describe().show()

    log("Merged DataFrames!")
    merged_df
  }

  private def csvToDataframe(spark:SparkSession, filename:String, logging:Boolean):DataFrame = {
    log("Parsing '" + filename + "' file...")
    var df = spark.read.format("csv")
      .option("header", "true")
      .load("resources/" + filename)

    if (logging) {
      println(df.show(20, truncate = 80))
      //df.printSchema()
      //df.describe().show()
      log("Parsing '" + filename + "' completed!")
    }

    df
  }

  private def flattenAttributes(spark:SparkSession, dataFrame: DataFrame, logging:Boolean):DataFrame = {
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._

    val flatten = dataFrame.withColumn("attr",concat_ws(" ",$"name" , $"value"))
                           .select("product_uid","attr")
                           .groupBy("product_uid")
                           .agg(concat_ws("\0",collect_list("attr")).alias("attributes"))

    if (logging) {
      println(flatten.show(20, truncate = 80))
      //flatten.printSchema()
      //flatten.describe().show()
      log("Grouping attributes completed!")
    }

    flatten
  }

  def saveParquet(dfToSave:DataFrame, parquetName: String):Unit = {
    log("Saving dataframe to '" + parquetName + "'...")
    dfToSave.write.parquet("resources/" + parquetName)
  }

  def loadParquet(spark:SparkSession, parquetName: String):DataFrame = {
    log("Loading data from '" + parquetName + "'...")
    val df = spark.read.parquet("resources/" + parquetName)
    df.show(10)
    df
  }

  def log(message:String) = {
    println(Console.BLUE + new SimpleDateFormat("yy/MM/dd  hh:mm:ss").format(new Date()) + " > " + message + Console.BLACK)
  }
}
