import os
import pickle
from tqdm import trange
import requests
import datetime
import pandas_datareader.data  as web
import datetime
# import matplotlib.pyplot as plt
# from matplotlib.pylab import date2num
# from matplotlib.font_manager import FontProperties
# import mplfinance as mpf
# import matplotlib
import numpy as np
import pandas as pd

# def get_daily_historic_data_yahoo(
#         ticker, start_date=(2000,1,1),
#         end_date=datetime.date.today().timetuple()[0:3]
#     ):
#     """
#     Obtains data from Yahoo Finance returns and a list of tuples.
#
#     ticker: Yahoo Finance ticker symbol, e.g. "GOOG" for Google, Inc.
#     start_date: Start date in (YYYY, M, D) format
#     end_date: End date in (YYYY, M, D) format
#     """
#     # Construct the Yahoo URL with the correct integer query parameters
#     # for start and end dates. Note that some parameters are zero-based!
#     ticker_tup = (
#         ticker, start_date[1]-1, start_date[2],
#         start_date[0], end_date[1]-1, end_date[2],
#         end_date[0]
#     )
#     yahoo_url = "http://ichart.finance.yahoo.com/table.csv"
#     yahoo_url += "?s=%s&a=%s&b=%s&c=%s&d=%s&e=%s&f=%s"
#     yahoo_url = yahoo_url % ticker_tup
#
#     # Try connecting to Yahoo Finance and obtaining the data
#     # On failure, print an error message.
#     try:
#         yf_data = requests.get(yahoo_url).text.split("\n")[1:-1]
#         prices = []
#         for y in yf_data:
#             p = y.strip().split(',')
#             prices.append(
#                 (datetime.datetime.strptime(p[0], '%Y-%m-%d'),
#                 p[1], p[2], p[3], p[4], p[5], p[6])
#             )
#     except Exception as e:
#         print("Could not download Yahoo data: %s" % e)
#     return prices


def get_data_from_yahoo(ticker='', start=datetime.datetime(2000, 1, 1), end=datetime.datetime(2019, 7, 1)):
    # start = datetime.datetime(2005, 1, 1), end = datetime.datetime(2018, 10, 2)
    prices = web.get_data_yahoo(ticker, start, end)
    return prices


if __name__ == '__main__':
    # df = get_daily_historic_data_yahoo('AAPL')

    tickers = []
    pwd = os.path.dirname(__file__)
    tickers_txt = 'dow_jones_30_ticker.txt'
    df_tickers_pk_path = os.path.join(pwd, 'df_tickes.pk')

    dji_csv_path = os.path.join(pwd, '^DJI.csv')

    if not os.path.exists(df_tickers_pk_path):
        f = open(tickers_txt, 'r')
        for line in f.readlines():
            tickers.append(line[:-1])
        f.close()

        df_data = []
        for i in trange(len(tickers)):
            try:
                df_ticker = get_data_from_yahoo(tickers[i])
                df_ticker['tic'] = tickers[i]
                df_data.append(df_ticker)

            except Exception as e:
                print(e)

        print("end")


        f = open(df_tickers_pk_path, 'wb')
        pickle.dump(df_data, f)
        f.close()
    else:
        f = open(df_tickers_pk_path, 'rb')
        df_data = pickle.load(f)
        f.close()

    if not os.path.exists(dji_csv_path):
        df_ticker = get_data_from_yahoo('^DJI')
        df_ticker.to_csv(dji_csv_path)

    dow_30_path = os.path.join(pwd, 'dow_30.csv')
    pd.concat(df_data, axis=0).to_csv(dow_30_path)