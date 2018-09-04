import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.collection.mutable
import org.apache.spark.rdd.RDD


object Airline_Sentiment_Analysis {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Need two parameters ")
    }
    val spark = SparkSession
      .builder()
      .appName("Airline_Sentiment_Analysis ")
      .getOrCreate()

    val sc = spark.sparkContext
    import spark.implicits._

    val tweets = spark.read.option("header","true").csv(args(0))

    //Loading and cleaning of data
    val cleanedTweets = tweets.filter("text is not null")

    var output = ""

    // Label-assignment
    val calculateRatings = udf { (sentiment: String) =>
      if (sentiment.toLowerCase  == "positive") 5.0
      else if (sentiment.toLowerCase  == "neutral" ) 2.5
      else 1.0
    }
    val tweetsWithRatings = cleanedTweets.withColumn("ratings", calculateRatings($"airline_sentiment"))

    val airlineRatings = tweetsWithRatings.groupBy("airline").avg("ratings").toDF("Airline","Avg_Ratings")

    output = "Airline Average Ratings \n\n"
    airlineRatings.collect().map(x => output+=x +"\n")

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    val worstAirline = airlineRatings.orderBy(asc("Avg_Ratings")).first().getString(0)
    output += "\nWorst Airline: "+ worstAirline
    val bestAirline = airlineRatings.orderBy(desc("Avg_Ratings")).first().getString(0)
    output += "\nBest Airline: "+ bestAirline

    val worstAirlineTweets = cleanedTweets.filter($"airline" === worstAirline)
    val bestAirlineTweets = cleanedTweets.filter($"airline" === bestAirline)

    val tokenizedWorstAirlineTweets = tokenizer.transform(worstAirlineTweets)
    val tokenizedBestAirlineTweets = tokenizer.transform(bestAirlineTweets)

    val filteredWorstAirlineTweets = stopWordsRemover.transform(tokenizedWorstAirlineTweets)
    val filteredBestAirlineTweets = stopWordsRemover.transform(tokenizedBestAirlineTweets)

    val filteredWorstAirlineTweets2 = filteredWorstAirlineTweets.withColumn("tweets", concat_ws(" ", $"filtered"))
    val filteredBestAirlineTweets2 = filteredBestAirlineTweets.withColumn("tweets", concat_ws(" ", $"filtered"))

    val worstAirlineTweetCorpus = filteredWorstAirlineTweets2.select(concat_ws(" ",collect_list(filteredWorstAirlineTweets2("tweets"))).alias("tweets")).first().getString(0)
    val bestAirlineTweetCorpus = filteredBestAirlineTweets2.select(concat_ws(" ",collect_list(filteredBestAirlineTweets2("tweets"))).alias("tweets")).first().getString(0)

    val worstAirlineTweetsRDD = sc.parallelize(Seq(worstAirlineTweetCorpus))
    val bestAirlineTweetsRDD = sc.parallelize(Seq(bestAirlineTweetCorpus))

    val worstAirlineTokenizedTweets: RDD[Seq[String]] = worstAirlineTweetsRDD.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 2).filter(_.forall(java.lang.Character.isLetter)))
    val bestAirlineTokenizedTweets: RDD[Seq[String]] = bestAirlineTweetsRDD.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 2).filter(_.forall(java.lang.Character.isLetter)))

    val worstAirlineTweetTermCounts: Array[(String, Long)] = worstAirlineTokenizedTweets.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)
    val bestAirlineTweetTermCounts: Array[(String, Long)] = bestAirlineTokenizedTweets.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)

    val worstAirlineTweetVocabArray: Array[String] = worstAirlineTweetTermCounts.takeRight(worstAirlineTweetTermCounts.size).map(_._1)
    val bestAirlineTweetVocabArray: Array[String] = bestAirlineTweetTermCounts.takeRight(bestAirlineTweetTermCounts.size).map(_._1)

    val worstAirlineTweetsVocab: Map[String, Int] = worstAirlineTweetVocabArray.zipWithIndex.toMap
    val bestAirlineTweetsVocab: Map[String, Int] = bestAirlineTweetVocabArray.zipWithIndex.toMap

    val worstAirlineTweetDocuments: RDD[(Long, Vector)] =
      worstAirlineTokenizedTweets.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (worstAirlineTweetsVocab.contains(term)) {
            val idx = worstAirlineTweetsVocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(worstAirlineTweetsVocab.size, counts.toSeq))
      }

    val bestAirlineTweetDocuments: RDD[(Long, Vector)] =
      bestAirlineTokenizedTweets.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (bestAirlineTweetsVocab.contains(term)) {
            val idx = bestAirlineTweetsVocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(bestAirlineTweetsVocab.size, counts.toSeq))
      }


    val numTopics = 5
    val airlineLDA = new LDA().setK(numTopics).setMaxIterations(100).setOptimizer("em")

    val worstAirlineLDAModel = airlineLDA.run(worstAirlineTweetDocuments)
    // Print topics, showing top-weighted 10 terms for each topic.
    output += "\n\n5 Topics for Worst Airline: \n\n";
    val worstAirlineTopicIndices = worstAirlineLDAModel.describeTopics(maxTermsPerTopic = 10)
    worstAirlineTopicIndices.foreach { case (terms, termWeights) =>
      output += "TOPIC: \n"
      terms.zip(termWeights).foreach { case (term, weight) =>
        output = output+""+ worstAirlineTweetVocabArray(term.toInt)+"\t"+weight+"\n"
      }
      output += "\n"
    }
    val bestAirlineLDAModel = airlineLDA.run(bestAirlineTweetDocuments)
    // Print topics, showing top-weighted 10 terms for each topic.
    output += "\n\n5 Topics for Best Airline: \n\n";
    val bestAirlineTopicIndices = bestAirlineLDAModel.describeTopics(maxTermsPerTopic = 10)
    bestAirlineTopicIndices.foreach { case (terms, termWeights) =>
      output += "TOPIC: \n"
      terms.zip(termWeights).foreach { case (term, weight) =>
        output = output+""+ bestAirlineTweetVocabArray(term.toInt)+"\t"+weight+"\n"
      }
      output += "\n"
    }

    sc.parallelize(List(output)).saveAsTextFile(args(1))

    sc.stop()

  }

}