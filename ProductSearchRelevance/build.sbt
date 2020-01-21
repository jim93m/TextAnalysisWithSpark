name := "ProductSearchRelevance"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.3.0"

val sparkStemmingVersion = "0.1.1"

val sparkMlVersion = "2.4.4"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "master" % "spark-stemming" % sparkStemmingVersion,
  "org.apache.spark" %% "spark-mllib" % sparkMlVersion
)

resolvers ++= Seq(
  "spark-stemming" at "https://dl.bintray.com/spark-packages/maven/"
)
