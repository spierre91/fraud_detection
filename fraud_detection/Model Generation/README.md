# solidus-ml-analytics
Collecting data from binance (requestbinance.py)
This script retreived data from the binance exchange. When run it prompts the user for a coin pair
and time period. The output is a json file with (named: Binanceagg_SYMBOL_date_to_ms.json)

Calculating features (getfeatures.py)
This script calculates the features that will be used as input in the KNN algo. The features are
the maximum return over of 30 min time period in the historical trade data, the maximum percent change in volume 
over a 30 min period and the volatility in price. 

Storing features (featureset.py)
the features are written to a json file for each coin pair

getREQpython.py makes HTTP get requests to influxDB on the AWS server and formats the data and stores bids/asks prices and sizes in a pandas data frame.
