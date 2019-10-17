# solidus-ml-analytics

CONFIGURATION
window_assumptions:
-contains json files with window sample size, sampling steps sizes for finding local minima
database_config:
-contains json with database parameters: AWS url, port, influx username, influx password, database name, and chunksize for query chunks
data multiple times instead of single large queries 
model_tuning:
-contains the number of KNN neighbors, sensityvity user input for pump and dump detection, the threshold for the magnitude of dumps,
and an accuracy parameter to run train test split several times for model scoring 

TESTS
-unit_tests: contains methods for testing resampling of volume, returns, trades and volatility.
-integration_test: contains several test cases for pump and dump and normal market behavior 
-test_knn_method: contains a series of test with different values for feature inputs [returns, volatility, volume, trades]. This test for large feature input, small feature input and for large volume (crypto whale activity)


KNN ALGORITHM
-contains the main k-nearest neighbors algorithm. 

MODEL GENERATION 
contains relevant scripts and files used during explarotory data analysis and model generation. 


SPOOFING
Contains gaussian model for outlier detection in order book data. The features are matched and cancelled volumes from one-minute snapshots of order book data. The model is based on Andrew Ng's anomaly detection lectures:https://www.coursera.org/learn/machine-learning. The code is explained in detail here: http://aqibsaeed.github.io/2016-07-17-anomaly-detection/. 

DOCUMENTATION
Contains documentation for spoofing and pump and dump algorithms. 
