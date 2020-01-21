package gr.auth.csd.dws

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.sql.{DataFrame, SparkSession}

object IOUtils {

  def parseAndJoinCSVs(spark: SparkSession):DataFrame = {
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
    val trainDF = csvToDataframe(spark, "train.csv", logging = true)
    val productDescriptionsDF = csvToDataframe(spark, "product_descriptions.csv", logging = true)
    val attributesDF = csvToDataframe(spark, "attributes.csv", logging = true)

    log("Merging Dataframes...")

    val tmp_df = trainDF.join(productDescriptionsDF, usingColumns = Seq("product_uid"), joinType = "left")
    val merged_df = tmp_df.join(attributesDF, usingColumns = Seq("product_uid"), joinType = "left")

    println(merged_df.show(10))
    merged_df.printSchema()
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
      println(df.show(20))
      df.printSchema()
      //df.describe().show()
      log("Parsing 'train.csv' completed!")
    }

    df
  }

  def log(message:String) = {
    println(Console.BLUE + new SimpleDateFormat("yy/MM/dd  hh:mm:ss").format(new Date()) + " > " + message + Console.BLACK)
  }

  // Testing object creation
  //    var product1 = new Product("100001", "Simpsons Light Tie", "Lorem Ipsum...")
  //    var product3 = new Product("100003", "Simpsons Strong Tie", "Lorem IpsumLorem IpsumLorem IpsumLorem Ipsum...")
  //    var product2 = new Product("100002", "Simpsons Regular Tie", "Lorem IpsumIpsumIpsumIpsum...")
  //
  //    var search1 = new Search(1, "simpsons", List(product1, product2))
  //    var search2 = new Search(2, "simpsonss", List(product2, product3, product1))
  //    var search3 = new Search(3, "simpsonsqwqw", List(product2))
  //
  //    var relevance1 = new Relevance(search1, product1, 3)
  //    var relevance2 = new Relevance(search2, product1, 2)
  //    var relevance3 = new Relevance(search3, product3, 1)
  //
  //
  //    product1.printProductDescrption
  //    product2.printProductDescrption
  //    product3.printProductDescrption
  //
  //    search1.printListOfProducts
  //
  //    println(relevance1.isRelevant())
  //    println(relevance2.isRelevant())
  //    println(relevance3.isRelevant())



  //    val conf = new SparkConf().setAppName("Product Search Relevance").setMaster("local")
  //    val sc = new SparkContext(conf)
  //    val textFile = sc.textFile("resources/testtt.txt")
  //    val counts = textFile.flatMap(line => line.split(" "))
  //                    .map(word => (word, 1))
  //                    .reduceByKey(_ + _)
  //    counts.collect().foreach(word => println(word))
}
