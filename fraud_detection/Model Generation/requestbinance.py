#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:22:36 2018

@author: spierre91
"""

import time
import dateparser
import pytz
import json

from datetime import datetime
from binance.client import Client
from binance import client
import pandas as pd


api_key='Tpxx7AASXQpZaTum02n50Zd8YRDfrcqB7GQ784BdO1v1obTKj4cfenKxvUdfYl9e'
api_secret=' qYMqBc108eJqGUJhezPO8Qt3ieErnmojybM7FYfoB10E9rZDnGbX4oDzHw4nlzzJ'
client = Client(api_key, api_secret)

def date_to_milliseconds(date_str):
    """Convert UTC date to milliseconds
    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/
    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)


def interval_to_milliseconds(interval):
    """Convert a Binance interval string to milliseconds
    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str
    :return:
         None if unit not one of m, h, d or w
         None if string not in correct format
         int value of interval in milliseconds
    """
    ms = None
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms

def get_aggregate_trades(self, **params):
        """Get compressed, aggregate trades. Trades that fill at the time,
        from the same order, with the same price will have the quantity aggregated.
        https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md#compressedaggregate-trades-list
        :param symbol: required
        :type symbol: str
        :param fromId:  ID to get aggregate trades from INCLUSIVE.
        :type fromId: str
        :param startTime: Timestamp in ms to get aggregate trades from INCLUSIVE.
        :type startTime: int
        :param endTime: Timestamp in ms to get aggregate trades until INCLUSIVE.
        :type endTime: int
        :param limit:  Default 500; max 500.
        :type limit: int
        :returns: API response
        .. code-block:: python
            [
                {
                    "a": 26129,         # Aggregate tradeId
                    "p": "0.01633102",  # Price
                    "q": "4.70443515",  # Quantity
                    "f": 27781,         # First tradeId
                    "l": 27781,         # Last tradeId
                    "T": 1498793709153, # Timestamp
                    "m": true,          # Was the buyer the maker?
                    "M": true           # Was the trade the best price match?
                }
            ]
        :raises: BinanceRequestException, BinanceAPIException
        """
        return self._get('aggTrades', data=params)


def aggregate_trade_iter(self, symbol, start_str=None, last_id=None):
        """Iterate over aggregate trade data from (start_time or last_id) to
        the end of the history so far.
        If start_time is specified, start with the first trade after
        start_time. Meant to initialise a local cache of trade data.
        If last_id is specified, start with the trade after it. This is meant
        for updating a pre-existing local trade data cache.
        Only allows start_str or last_id—not both. Not guaranteed to work
        right if you're running more than one of these simultaneously. You
        will probably hit your rate limit.
        See dateparser docs for valid start and end string formats http://dateparser.readthedocs.io/en/latest/
        If using offset strings for dates add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
        :param symbol: Symbol string e.g. ETHBTC
        :type symbol: str
        :param start_str: Start date string in UTC format or timestamp in milliseconds. The iterator will
        return the first trade occurring later than this time.
        :type start_str: str|int
        :param last_id: aggregate trade ID of the last known aggregate trade.
        Not a regular trade ID. See https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md#compressedaggregate-trades-list.
        :returns: an iterator of JSON objects, one per trade. The format of
        each object is identical to Client.aggregate_trades().
        :type last_id: int
        """
        if start_str is not None and last_id is not None:
            raise ValueError(
                'start_time and last_id may not be simultaneously specified.')

        # If there's no last_id, get one.
        if last_id is None:
            # Without a last_id, we actually need the first trade.  Normally,
            # we'd get rid of it. See the next loop.
            if start_str is None:
                trades = self.get_aggregate_trades(symbol=symbol, fromId=0)
            else:
                # The difference between startTime and endTime should be less
                # or equal than an hour and the result set should contain at
                # least one trade.
                if type(start_str) == int:
                    start_ts = start_str
                else:
                    start_ts = date_to_milliseconds(start_str)
                trades = self.get_aggregate_trades(
                    symbol=symbol,
                    startTime=start_ts,
                    endTime=start_ts + (60 * 60 * 1000))
            for t in trades:
                yield t
            last_id = trades[-1][self.AGG_ID]

        while True:
            # There is no need to wait between queries, to avoid hitting the
            # rate limit. We're using blocking IO, and as long as we're the
            # only thread running calls like this, Binance will automatically
            # add the right delay time on their end, forcing us to wait for
            # data. That really simplifies this function's job. Binance is
            # fucking awesome.
            trades = self.get_aggregate_trades(symbol=symbol, fromId=last_id)
            # fromId=n returns a set starting with id n, but we already have
            # that one. So get rid of the first item in the result set.
            trades = trades[1:]
            if len(trades) == 0:
                return
            for t in trades:
                yield t
            last_id = trades[-1][self.AGG_ID]
            


start =input("Input time period. Example: ' 3 min ago UTC':")
symbol=input("Input coin pair. Example: for LTC/BTC input LTCBTC:")

   
def writeJSON(start,symbol):
    agg_trades = client.aggregate_trade_iter(symbol, start)
    agg_trade_list = list(agg_trades)
    
    with open(
        "Binanceagg_{}-{}.json".format(
            symbol,
            date_to_milliseconds(start),
        ),
        'w'  # set file write mode
    ) as f:
        f.write(json.dumps(agg_trade_list))


writeJSON(start,symbol)
