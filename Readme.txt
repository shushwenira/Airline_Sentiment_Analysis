
Steps to run the Application:

---------------------------------------------------------------------------------------------------------------------------------------------------------------

Upload the Jar and the coresponding kaggle (https://www.kaggle.com/crowdflower/twitter-airline-sentiment) csv data into a s3 bucket.

Run AWS Elastic Map Reduce Step with Following configuration:

---------------------------------------------------------------------------------------------------------------------------------------------------------------

Step type: Spark Application
Name: Airline_Sentiment_Analysis
Deploy mode: Cluster
Spark-submit options: --class "Airline_Sentiment_Analysis"
Application location*: s3://your-bucket/Airline_Sentiment_Analysis_2.11-0.1.jar
Arguments: s3://your-bukcet/Tweets.csv 
	   s3://your-bukcet/output-folder
Action on failure: Continue

---------------------------------------------------------------------------------------------------------------------------------------------------------------

Output will be present in your specified s3 location 

---------------------------------------------------------------------------------------------------------------------------------------------------------------
