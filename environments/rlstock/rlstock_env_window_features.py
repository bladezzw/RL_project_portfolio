import os
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import pickle
import matplotlib.pyplot as plt

data_dir = 'Data_Daily_Stock_Dow_Jones_30'
file_name = 'dow_jones_30_daily_price.csv'
abs_data_dirname = os.path.join(os.path.dirname(__file__), data_dir)
abs_data_path = os.path.join(abs_data_dirname, file_name)
abs_rlstock_env_window = os.path.join(abs_data_dirname, 'Traindata_rlstock_env_window_features.pk')


# df = pd.read_csv('/home/zzw/pywork_2020//DQN-DDPG_Stock_Trading-master/venv/lib/python3.6/site-packages/gym/envs/rlstock/Data_Daily_Stock_Dow_Jones_30/dow_jones_30_daily_price.csv')
df = pd.read_csv(abs_data_path)


f = open(abs_rlstock_env_window, 'rb')
train_daily_data = pickle.load(f)
f.close()

dji_file_name = '^DJI.csv'
abs_data_path_dji = os.path.join(abs_data_dirname, dji_file_name)
dji = pd.read_csv(abs_data_path_dji)
def dji_func(start_date = '2016-01-01', end_date='2020-01-01'):
    test_dji = dji[(dji['Date'] > start_date) & (dji['Date']< end_date)]
    # test_dji = dji[dji['Date'] >= '2005-01-01']
    dji_price = test_dji['Adj Close']
    dji_date = test_dji['Date']
    daily_return = dji_price.pct_change(1)
    daily_return = daily_return[1:]
    daily_return.reset_index()
    initial_amount = 10000

    total_amount = initial_amount
    account_growth = list()
    account_growth.append(initial_amount)
    for i in range(len(daily_return)):
        total_amount = total_amount * (daily_return.iloc[i]+1)  # + total_amount
        account_growth.append(total_amount)
    return account_growth
account_growth = dji_func('2005-01-01', '2016-01-01')


class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, day=0, money=10000, scope=1, action_high=1, action_shape=27,**kwargs):
        self.name = 'Stock_portfolio_v1'
        self.day = day
        self.money = money
        # buy or sell maximum 5 shares
        self.action_shape = action_shape
        self.action_space = spaces.Box(low=0, high=action_high, shape=(self.action_shape+1,), dtype=np.float)

        self.train_daily_data = train_daily_data['obs']
        self.Date = train_daily_data['dates']
        self.prices = train_daily_data['prices']

        self.data = self.train_daily_data[self.day]

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(*self.data.shape, ))

        self.terminal = False

        self.state = [self.data, np.array([1] + [0 for i in range(self.action_shape)])]
        self.reward = 0

        self.asset_memory = [self.money]

        self.reset()
        self._seed()

    def __str__(self):
        return self.name

    def plot(self, iter=None):
        iter = iter if iter else ''

        try:
            plt.plot(self.asset_memory, 'r')
            if account_growth:
                plt.plot(account_growth)

            plt.savefig('result_training%s.png'% iter)
            plt.close()
        except:
            f = open("result_training%s.pk"% iter, 'wb')
            pickle.dump([self.asset_memory, account_growth], f)
            f.close()

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.train_daily_data) - 1
        # print(actions) n

        if self.terminal:

            self.plot()

            print("total_reward:{}".format(self.asset_memory[-1]))

            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            return self.state, self.reward, self.terminal, {}

        else:
            a_t_1 = self.state[1]
            self.day += 1
            self.data = self.train_daily_data[self.day]
            self.state = [self.data, actions]

            y_t = np.insert(self.prices[self.day]/self.prices[self.day-1], 0, 1.)


            y_t_w_t_1 = (y_t*a_t_1).sum()

            self.reward = np.log(y_t_w_t_1)

            p_t = self.asset_memory[-1]*y_t_w_t_1

            self.asset_memory.append(p_t)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.money]
        self.day = 0
        # self.data = train_daily_data[self.day]
        self.data = self.train_daily_data[self.day]
        # self.state = [self.money] + self.data.Close.values.tolist() + [0 for i in range(self.action_shape)]
        self.state = [self.data, np.array([1] + [0 for i in range(self.action_shape)])]
        # iteration += 1
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

if __name__ == "__main__":

    env = StockEnv()
    o = env.reset()
    done = 0
    while not done:
        # a = env.action_space.sample()
        a = np.array([1 for i in range(28)])
        a = np.exp(a)/sum(np.exp(a))
        o, r, done, _ = env.step(a)

    print("end")