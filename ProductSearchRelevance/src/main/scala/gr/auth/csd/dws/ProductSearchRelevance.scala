package gr.auth.csd.dws

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level
import gr.auth.csd.dws.IOUtils._
import org.apache.commons.lang.time.DurationFormatUtils


object ProductSearchRelevance {

  def main(args: Array[String]): Unit = {
    val timeStarted = System.currentTimeMillis()
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val appName = "Product Search Relevance"


    log("Initializing " + appName)
    log("Init Spark Context")
    val spark = SparkSession.builder.master("local").appName(appName).getOrCreate



    log("CSVs to merged DF")
    var df = IOUtils.parseAndJoinCSVs(spark, logging = false)
//    saveParquet(df, "merged_CSVs.df")
    //or
//    var df = loadParquet(spark, "merged_CSVs.df")


    log("Data sampling...")
    log("Initial DataFrame size: " + df.count())
    df = df.sample(0.02, 323)
    log("Sample size: " + df.count())




    log("DataFrame Pre-processing...")
    val pre_processed_df = Preprocess.run(spark, df, logging = false)
//    saveParquet(pre_processed_df, "preprocessed.df")
    //or
//    val pre_processed_df = loadParquet(spark, "preprocessed.df")

    log("Generating Features...")
    val final_df = FeatureFactory.generateFeatures(spark, pre_processed_df)
    final_df.show(10)

    val neededCols_df = Preprocess.dropUnneededColumns(spark, final_df)
    val ml_df = Preprocess.renameColumns(neededCols_df)
    ml_df.show(10, truncate = 80)

//    saveParquet(ml_df, "ML_df.df")
//    val ml_df = loadParquet(spark, "ML_df.df")




    //ML stuff
    //Random Forests model
    // Gradient Boosting model and
    // Xgboost model






    val timeFinished = System.currentTimeMillis()
    log("Total time: " + DurationFormatUtils.formatDuration(timeFinished - timeStarted, "HH:mm:ss"))
  }


}
