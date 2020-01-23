package gr.auth.csd.dws

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

import gr.auth.csd.dws.IOUtils.log


object ProductSearchRelevance {

  def main(args: Array[String]): Unit = {
    //System.setProperty("hadoop.home.dir", "C:/hadoop/bin")
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val appName = "Product Search Relevance"


    log("Initializing " + appName)
    log("Init Spark Context")
    val spark = SparkSession.builder.master("local").appName(appName).getOrCreate

    log("CSVs to merged DF")
    var df = IOUtils.parseAndJoinCSVs(spark)

//    log("Saving merged dataframe parquet...")
//    df.write.parquet("resources/merged_CSVs.df")
//    log("Loading merged dataframe parquet...")
//    var df = spark.read.parquet("resources/merged_CSVs.df")
//    df.show(10)


    //Data sampling
    log("Initial DataFrame size: " + df.count())
    df = df.sample(0.1, 323)
    log("Sample size: " + df.count())
    df.describe().show()




    log("DataFrame Pre-processing...")
    val pre_processed_df = Preprocess.stemming(df)

    log("Generating Features...")
    val final_df = FeatureFactory.generateFeatures(spark, pre_processed_df)
    final_df.show(10)


//    log("Saving final dataframe parquet...")
//    final_df.write.parquet("resources/final_df.df") //parquet

//    log("Loading final dataframe parquet...")
//    val final_df = spark.read.parquet("resources/final_df.df")
//    final_df.show(10)




    //ML stuff
    //Random Forests model
    // Gradient Boosting model and
    // Xgboost model







  }


}
