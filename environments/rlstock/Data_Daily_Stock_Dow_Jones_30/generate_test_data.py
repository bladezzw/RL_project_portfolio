import os
import numpy as np
import pandas as pd
from tqdm import trange
import pickle

file_name = 'dow_jones_30_daily_price.csv'
abs_data_dirname = os.path.dirname(__file__)
abs_data_path = os.path.join(abs_data_dirname, file_name)
abs_rlstock_env_window = os.path.join(abs_data_dirname, 'Testdata_rlstock_env_window_features.pk')
dji_file_name = 'DJI_growth.pk'
abs_dji_path = os.path.join(abs_data_dirname, dji_file_name)
# df = pd.read_csv('/home/zzw/pywork_2020//DQN-DDPG_Stock_Trading-master/venv/lib/python3.6/site-packages/gym/envs/rlstock/Data_Daily_Stock_Dow_Jones_30/dow_jones_30_daily_price.csv')
df = pd.read_csv(abs_data_path)

data_1 = df.copy()
equal_4905_list = list(data_1.tic.value_counts() == 4905)
names = data_1.tic.value_counts().index

# select_stocks_list = ['NKE','KO']
# select_stocks_list = list(names[equal_4905_list]) + ['NKE', 'KO']
select_stocks_list = list(names[equal_4905_list])

# data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912', '20010913'])]
data_2 = data_1[data_1.tic.isin(select_stocks_list)]
# data_3 = data_2[['iid', 'datadate', 'tic', 'prccd', 'prcld', 'prcod', 'prchd', 'ajexdi']]
data_3 = data_2[['Date', 'tic', 'High', 'Low', 'Open', 'Close', 'Adj Close']]

data_3['High'] = data_3['High'] * data_3['Adj Close'] / data_3['Close']
data_3['Low'] = data_3['Low'] * data_3['Adj Close'] / data_3['Close']
data_3['Open'] = data_3['Open'] * data_3['Adj Close'] / data_3['Close']
data_3['Close'] = data_3['Close'] * data_3['Adj Close'] / data_3['Close']
del data_3['Adj Close'], data_3['Open']

start_date = '2016-01-01'
train_data = data_3[(data_3.Date > start_date)]


from generate_trainning_data import generate_data
# def generate_data(train_data=train_data, windows=5):
#     tics_data = []
#     for tic in select_stocks_list:
#         tics_data.append(train_data[train_data.tic == tic].reset_index())
#
#     train_daily_data = []
#     Date = np.unique(train_data.Date)
#     dates = []
#     prices = []
#     for j in trange(len(Date)):
#         date = Date[j]
#         if date >= Date[windows - 1]:
#             data_list = []
#
#             price_iloc = []
#             for i in range(len(select_stocks_list)):
#                 temp_tics_data = tics_data[i]
#                 iloc = temp_tics_data[temp_tics_data.Date == date].index.values[0]
#
#                 price_iloc.append(temp_tics_data.Close.iloc[iloc])
#                 obs_one = temp_tics_data.Close.iloc[iloc] - np.array(
#                     temp_tics_data.iloc[iloc - windows + 1:iloc + 1, -3:])
#                 obs_one = obs_one / obs_one.std()
#                 data_list.append(obs_one)
#
#             train_daily_data.append(np.array(data_list))
#             dates.append(date)
#             prices.append(np.array(price_iloc))
#
#     x = np.array(train_daily_data)
#     di_ = dict(obs=x,
#                dates=dates,
#                prices=np.array(prices))
#     # for i in range(len(x)-windows):  # 检查是否一致
#     #     if tics_data[0].iloc[i,-1] - x[i,0,0,-1] > 0.01:
#     #         print(1)
#     if not os.path.exists(abs_rlstock_env_window):
#         f = open(abs_rlstock_env_window, 'wb')
#         pickle.dump(di_, f)
#         f.close()
#
#     print('end')


if __name__ == "__main__":
    generate_data(train_data, filename=abs_rlstock_env_window)
