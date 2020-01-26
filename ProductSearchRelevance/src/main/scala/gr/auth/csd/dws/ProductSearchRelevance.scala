package gr.auth.csd.dws

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level
import gr.auth.csd.dws.IOUtils._
import org.apache.commons.lang.time.DurationFormatUtils
import org.apache.spark.ml.clustering.{BisectingKMeans, KMeans}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression, RandomForestRegressor}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions.{col, udf}


object ProductSearchRelevance {
  var binaryThreshold: Double = 2.0

  def main(args: Array[String]): Unit = {
    val timeStarted = System.currentTimeMillis()
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val appName = "Product Search Relevance"


    log("Initializing " + appName)
    log("Init Spark Context")
    val spark = SparkSession.builder.master("local").appName(appName).getOrCreate
    import spark.implicits._


    log("CSVs to merged DF")
    var df = IOUtils.parseAndJoinCSVs(spark, logging = false)
//    saveParquet(df, "merged_CSVs.df")
    //or
//    var df = loadParquet(spark, "merged_CSVs.df")


    log("Data sampling...")
    log("Initial DataFrame size: " + df.count())
    df = df.sample(0.1, 323)
    log("Sample size: " + df.count())




    log("DataFrame Pre-processing...")
    val pre_processed_df = Preprocess.run(spark, df, logging = false)
//    saveParquet(pre_processed_df, "preprocessed.df")
    //or
//    val pre_processed_df = loadParquet(spark, "preprocessed.df")

    log("Generating Features...")
    val feature_df = FeatureFactory.generateFeatures(spark, pre_processed_df)
    feature_df.show(10)

    val neededCols_df = Preprocess.dropUnneededColumns(spark, feature_df)
    val ml_df = Preprocess.renameColumns(neededCols_df)
    ml_df.show(10, truncate = 80)

//    saveParquet(ml_df, "ML_df.df")
//    val ml_df = loadParquet(spark, "ML_df.df")

    import org.apache.spark.sql.functions._

    val toDouble = udf[Double, String]( _.toDouble)
    val feature_df_doubleCast = ml_df.withColumn("relevance",toDouble(df("relevance")))




    val final_df =(feature_df_doubleCast.select(feature_df_doubleCast("relevance").as("label"),
      $"len_of_query",$"levenshtein_dist_description",$"levenshtein_dist_title",$"levenshtein_dist_attr",$"len_of_attr"
      ,$"commonWords_term_title",$"commonWords_term_description",$"commonWords_term_attr",$"ratio_title",$"ratio_description"
      ,$"ratio_attr"
    ))
    val assembler = (new VectorAssembler().setInputCols(Array(
      "len_of_query","levenshtein_dist_description","levenshtein_dist_title","levenshtein_dist_attr","len_of_attr"
      ,"commonWords_term_title","commonWords_term_description","commonWords_term_attr","ratio_title"
      ,"ratio_description","ratio_attr" )).setOutputCol("features"))

    val output = assembler.setHandleInvalid("skip").transform(final_df).select($"label", $"features")
    output.show(10)


    val Array(training, test) = output.randomSplit(Array(0.9, 0.1), seed = 12345)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("mse")

    //Linear Regression
    val lr = new LinearRegression()
    val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(100,0.1))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()
    val trainValSplit = (new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator().setMetricName("mse"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.6))
    val lrModel = trainValSplit.fit(training)

    //val lrModel = lr.fit(training)
    val lrPredictions = lrModel.transform(test)

/*
    // Descritize predictions to certain levels
    // Not currently used, as it did not offer better results
    val roundToLevels = udf((pred: Double) => {
      val levels: List[Double] = List(1.0, 1.33, 1.67, 1.75, 2.0, 2.25, 2.33, 2.5, 2.67, 2.75, 3.0)
      var minimum: Double = 2.0
      for (num <- levels) {
        if ((num - pred).abs < (pred - minimum).abs) {
          minimum = num
        }
      }
      minimum
    })
    var lrPredictionsRounded = lrPredictions.withColumn("predRound", roundToLevels(col("prediction")))
    lrPredictionsRounded = lrPredictionsRounded.drop(col("prediction")).withColumnRenamed("predRound", "prediction")
*/
    println("Validation metric Mean Squared Error (MSE) for Linear Regression models on test data = " + lrModel.validationMetrics.mkString(" "))
    val lr_mse = evaluator.evaluate(lrPredictions)
    //val lr_mse = evaluator.evaluate(lrPredictionsRounded)
    println("Best model MSE on test data = " + lr_mse)



    //Random Forest Regressor

    val rf = new RandomForestRegressor()

    val paramGridRf = new ParamGridBuilder().addGrid(rf.numTrees, Array(10,15,20,25))
      .addGrid(rf.maxDepth, Array(2,4,6,8,10))
      .build()
    val trainValSplitRf = (new TrainValidationSplit()
      .setEstimator(rf)
      .setEvaluator(new RegressionEvaluator().setMetricName("mse"))
      .setEstimatorParamMaps(paramGridRf)
      .setTrainRatio(0.6))

    val rfModel = trainValSplit.fit(training)
    //val rfModel = rf.fit(training)

    val rfPredictions = rfModel.transform(test)

    println("Validation metric Mean Squared Error (MSE) for Random Forest Regressor models on test data = " + rfModel.validationMetrics.mkString(" "))
    val rf_mse = evaluator.evaluate(rfPredictions)
    println("Random Forest Regressor Mean Squared Error (MSE) on test data = " + rf_mse)





    //Gradient-Boosted Tree Regressor
    val gbt = new GBTRegressor().setLabelCol("label").setFeaturesCol("features")
    val gbtModel = gbt.fit(training)
    val gbtPredictions = gbtModel.transform(test)
    gbtPredictions.select("prediction", "label", "features").show(10)

    val gbt_mse = evaluator.evaluate(gbtPredictions)
    println("Gradient-Boosted Tree Regressor Mean Squared Error (MSE) on test data = " + gbt_mse)



    // Descritize predictions into two levels
    val roundTo2Levels = udf((label: Double) => {
      var binary: Double = 0.0
      if (label > binaryThreshold) {
        binary = 1.0
      }
      else {
        binary = 0.0
      }
      binary
    })
    //binaryThreshold = output.select(avg($"label")).first().getDouble(0) // Setting the binary threshold for the two clusters
    println("Binary Threshold: "+ binaryThreshold)
    var outputDisc = output.withColumn("labelRound", roundTo2Levels(col("label")))
    outputDisc.show()
    outputDisc = outputDisc.drop(col("label")).withColumnRenamed("labelRound", "label")
    outputDisc.show()


    println("Starting Kmeans ")
    val kmeans = new KMeans().setK(2).setSeed(1L)
    val kmeansModel = kmeans.fit(outputDisc)
    val kmeansPredictions = kmeansModel.transform(outputDisc)
    kmeansPredictions.show(20)

    val kmeansPredictionsAndLabels = kmeansPredictions.select($"prediction",$"label").as[(Double,Double)].rdd
    val metricsKmeans = new BinaryClassificationMetrics(kmeansPredictionsAndLabels)
    // F-measure
    val f1ScoreKmeans = metricsKmeans.fMeasureByThreshold
    val precisionKmeans = metricsKmeans.precisionByThreshold()
    val recallKmeans = metricsKmeans.recallByThreshold

    println("F1 Score for Kmeans is: ")
    f1ScoreKmeans.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }
    println("Precision for Kmeans is: ")
    precisionKmeans.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }
    println("Recall for Kmeans is: ")
    recallKmeans.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }


    //Bisecting k-means
    val bkm = new BisectingKMeans().setK(2).setSeed(1L)
    val bkmModel = bkm.fit(outputDisc)
    val bkmPredictions = bkmModel.transform(outputDisc)

    val bkmPredictionsAndLabels = bkmPredictions.select($"prediction",$"label").as[(Double,Double)].rdd
    val metricsBbk = new BinaryClassificationMetrics(bkmPredictionsAndLabels)
    // F-measure
    val f1ScoreBbk = metricsBbk.fMeasureByThreshold
    val precisionBbk = metricsBbk.precisionByThreshold()
    val recallBbk = metricsBbk.recallByThreshold

    println("F1 Score for Bkm is: ")
    f1ScoreBbk.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }
    println("Precision for Bkm is: ")
    precisionBbk.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }
    println("Recall for Bkm is: ")
    recallBbk.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }




  //  val timeFinished = System.currentTimeMillis()
  //  log("Total time: " + DurationFormatUtils.formatDuration(timeFinished - timeStarted, "HH:mm:ss"))
  }


}
