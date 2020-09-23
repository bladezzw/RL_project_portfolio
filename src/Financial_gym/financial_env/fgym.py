import os
import numpy as np
import copy
import gym
from gym.core import Env
from gym import spaces
from gym.utils import seeding
from collections import deque
from PIL import Image

class fin_env(Env):
    def __init__(self,
                 game='Finance',
                 mode=None,
                 data=None,
                 date=None,
                 seed=None,
                 data_shape=[],
                 windows=128,
                 ):
        """
        :param game: name of environment
        :param mode:
        :param data: (n, m)的np.ndarray矩阵，m为日期总数，n为特征总数
        :param seed: random seed
        :param data_shape: (n, m)
        :param obs_type:
        """
        self._action_set = np.ndarray([0, 1, 2])  # 0:卖, 1:不变, 2:买
        self.action_space = spaces.Discrete(len(self._action_set))
        self.seed()
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(data_shape[0], windows))

        pass

    def step(self, action):
        raise NotImplemented

    def reset(self):
        raise NotImplemented

    def render(self, mode='human'):
        raise NotImplemented

    def seed(self, seed=None):
        super(fin_env, self).seed(seed)


# class Normal_fin(Env):
#     ACTION_MEANING = {
#         -2: "大空,1).如有多头清空所有多头同时增加空头;2).如果没有多头,就增加空头",
#         -1: "空 , 1).如有多头清空所有多头后不买入;   2).如果没有多头,就增加空头",
#         0: "不变",
#         1: "多,   1).如有空头清空所有空头后不买入;   2).如果没有空头,就增加多头",
#         2: "大多, 1).如有空头清空所有空头同时增加多头;2).如果没有空头,就增加多头",
#     }
#     def __init__(self,
#                  game='Futures',
#                  mode=None,
#                  data=None,
#                  date=None,
#                  seed=None,
#                  data_shape=[],
#                  windows=128,
#                  warm_up=100000,
#                  init_capital=10000,
#                  show_withdraw=False,
#                  show_statistics=False,
#                  cost=-3.4,
#                 ):
#         """类似股票的,只能低买高卖才能盈利"""
#         self._action_set = [-2, -1, 0, 1, 2]  #
#         self.action_space = spaces.Discrete(len(self._action_set))
#         if data is None:
#             raise ValueError
#         self.data = data
#         self.date = date
#         self.Date = [d[:10] for d in self.date]
#         self.Time = [d[11:] for d in self.date]
#         self.Close = self.data[3, :]
#         if data_shape == []:
#             (self.data_n, self.data_m) = self.data.shape
#         self.windows = windows
#         self.warm_up = warm_up  # 预热
#         self.start_T = None  # reset()之后的开始时间索引
#         self.current_T = None  # 当前的时间索引
#         self.len_T = len(self.Time)
#         self.seed(seed)
#         self.observation_space = spaces.Box(low=-10000, high=10000, dtype=np.uint8, shape=(self.data_n, self.windows))
#
#         # financial affairs
#         self.init_capital = init_capital  # 初始资金
#         self.current_act_T = None  # 这次操作的时间索引
#         self.current_act_price = None  # 这次操作价格
#         self.act_cost = cost  # 每次操作的成本(对于股票来说卖出才有手续费)
#         self.returns = 0  # 收益
#         self.cash = copy.deepcopy(self.init_capital)  # 场外资金(现金)
#         self.value_in_market = 0  # 市场中的资金
#         self.total = 0  # 当前总资产
#         self.last_act_T = None  # 上次操作的时间索引,(操作有买入/卖出)
#         self.last_act_price = None  # 上次操作的价格
#
#         self.commission = 0
#         self.num_act = 1  # 每次操作的数量
#         self.direction = 0  # 持有的头寸方向
#         self.drawdown = None  # 回撤
#
#         self.show_withdraw = show_withdraw
#         self.show_statistics = show_statistics
#         self.max = copy.deepcopy(self.init_capital)
#
#     def _statistics(self):
#         raise NotImplemented
#
#     def _reward(self, action):
#         """这是股票的交易框架"""
#         if action == -1:  # 动作卖出
#             if self.direction == 1:  # 上次是已经买入了的
#                 self.direction = 0  # 卖出了，就没有场内头寸了，所以是0
#                 self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
#                 self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
#
#                 self.value_in_market = self.num_act*self.current_act_price  # 市场中的资金
#                 self.returns = 0  # 收益
#                 self.cash = self.cash - self.value_in_market - self.act_cost  # 场外资金(现金)
#
#                 self.total = self.cash + self.value_in_market  # 当前总资产
#                 self.last_act_T = self.current_act_T  # 上次操作的时间索引,(操作有买入/卖出)
#                 self.last_act_price = self.current_act_price  # 上次操作的价格
#
#                 self.commission += self.act_cost
#                 self.num_act = num_act  # 每次操作的数量
#                 self.direction = 0  # 持有的头寸方向
#                 self.drawdown = None  # 回撤
#
#
#
#                 r = self.current_act_price - self.last_act_price - self.act_cost  # 当前收益=上一个价格-当前操作价格
#
#                 self.returns += r
#                 self.cash = self.cash + self.current_act_price - self.act_cost
#                 self.value_in_market = 0  # 市场中的资金
#                 self.total = self.cash  # 当前总资产
#                 self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
#                 self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
#
#                 self.commission += self.act_cost
#                 if self.show_withdraw == True:
#                     if self.total > self.max:
#                         self.max = self.total
#                     self.drawdown = self.total/self.max  # 回撤
#
#                 if self.show_statistics == True:
#                     self._statistics()
#
#                 return r
#             elif self.direction == 0:
#                 pass
#
#             elif self.direction == -1:
#                 pass
#
#         elif action == 0:  # 动作不变
#             if self.direction == 1:  # 上次是已经买入了的
#                 self.value_in_market = self.Close[self.current_T]  # 市场中的资金
#                 self.total = self.cash + self.value_in_market  # 当前总资产
#                 r = 0
#
#                 if self.show_withdraw == True:  # 假设卖出才计算最高值,进而才计算回撤率
#                     pass
#
#                 if self.show_statistics == True:
#                     self._statistics()
#                 return r
#             elif self.direction == 0:
#                 pass
#
#             elif self.direction == -1:
#                 pass
#
#         elif action == 1:  # 动作买入
#             if self.direction == 0:  # 场内没有头寸
#                 self.direction = 1  # 买入头寸
#                 self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
#                 self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
#
#                 self.value_in_market = self.current_act_price
#
#                 self.cash = self.cash - self.value_in_market
#
#                 self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
#                 self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
#
#                 r = self.act_cost
#
#                 self.commission += self.act_cost
#                 if self.show_withdraw == True:
#                     if self.total > self.max:
#                         self.max = self.total
#                     self.drawdown = self.total / self.max  # 回撤
#
#                 if self.show_statistics == True:
#                     self._statistics()
#                 return r
#             elif self.direction == 1:
#                 pass
#
#             elif self.direction == -1:
#                 pass
#
#     def _done(self):
#         if self.total < self.Close[self.current_T]:  # 资金不足
#             self.close()
#             return 1
#         elif self.current_T == self.len_T:  # 达到索引中点
#             return 1
#         else:
#             return 0
#
#     def _info(self):
#         if self.show_statistics == True:
#             self._statistics()
#         return ''
#
#     def step(self, a):
#         """Returns:
#             observation (object): agent's observation of the current environment
#             reward (float) : amount of reward returned after previous action
#                 [如果reward是空,主函数中应跳到下一时刻,即再跳用一次step(.) ]
#             done (bool): whether the episode has ended, in which case further step() calls will return undefined results
#             info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
#         """
#         action = self._action_set[a]
#         reward = self._reward(action)
#         done = self._done()
#         info = self._info()
#
#         self.current_T += 1
#         observations = self.data[:, self.current_T - self.windows: self.current_T]
#         return observations, reward, done, info
#
#     def reset(self):
#         """
#         :return: observation (object): agent's observation of the current environment
#         """
#         self.start_T = self.np_random.randint(self.warm_up, self.len_T)
#         self.current_T = copy.deepcopy(self.start_T)
#         observations = self.data[:, self.start_T - self.windows: self.start_T]
#         return observations
#
#     def render(self, mode='human'):
#         raise NotImplemented
#
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]


class singal_fin_life(Env):  # UNTESTED
    ACTION_MEANING = {
        -2: "大空,1).如有多头清空所有多头同时增加空头;2).如果没有多头,就增加空头",
        -1: "空 , 1).如有多头清空所有多头后不买入;   2).如果没有多头,就增加空头",
        0: "不变",
        1: "多,   1).如有空头清空所有空头后不买入;   2).如果没有空头,就增加多头",
        2: "大多, 1).如有空头清空所有空头同时增加多头;2).如果没有空头,就增加多头",
    }

    def __init__(self,
                 game='Futures',
                 code="RB",
                 mode=None,
                 data=None,
                 date=None,
                 seed=None,
                 data_shape=[],
                 windows=128,
                 warm_up=100000,
                 init_capital=10000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=3.4,
                 signal_index=None,
                 num_act=1,
                 fold=10,
                 ):
        """面向期货的,能双向交易盈利
        :param game:
        :param code:
        :param mode: life or nodeath
        :param data:
        :param date:
        :param seed:
        :param data_shape:
        :param windows:
        :param warm_up:
        :param init_capital:
        :param show_withdraw:
        :param show_statistics:
        :param cost:
        :param signal_index: 必须指定信号是第几行.
        :param num_act: 执行的数量
        :param fold:
        """
        self._action_set = [-2, -1, 0, 1, 2]  # 0:卖, 1:不变, 2:买
        self.action_space = spaces.Discrete(len(self._action_set))
        self.mode = mode
        if data is None:
            raise ValueError
        self.data = data
        self.date = date
        self.Date = [d[:10] for d in self.date]
        self.Time = [d[11:] for d in self.date]
        self.Close = self.data[3, :]  # 不同数据可能所在的位置不一样
        if data_shape == []:
            (self.data_n, self.data_m) = self.data.shape
        self.windows = windows
        if warm_up is None:
            self.warm_up = round(self.data_m * 0.1)  # 预热
        else:
            self.warm_up = warm_up
        self.start_T = None  # reset()之后的开始时间索引
        self.current_T = None  # 当前的时间索引
        self.len_T = len(self.Time)
        assert self.len_T == self.data_m
        self.seed(seed)
        self.observation_space = spaces.Box(low=-10000, high=100000, dtype=np.uint8, shape=(self.data_n, self.windows))

        # financial affairs
        self.code = code
        self.init_capital = init_capital  # 初始资金
        self.fold = fold  # 每条代表的价格,如: 螺纹钢1跳为10元/手
        self.current_act_T = None  # 这次操作的时间索引
        self.current_act_price = None  # 这次操作价格
        self.returns = 0  # 收益
        self.cash = copy.deepcopy(self.init_capital)  # 场外资金(现金)
        self.value_in_market = 0  # 市场中的资金
        self.total = copy.deepcopy(self.init_capital)  # 当前总资产
        self.last_act_T = None  # 上次操作的时间索引,(操作有买入/卖出)
        self.last_act_price = None  # 上次操作的价格
        self.act_cost = cost  # 每次操作的成本(对于股票来说卖出才有手续费)
        self.commission = 0
        self.num_act = num_act  # 每次操作的数量
        self.direction = 0  # 持有的头寸方向
        self.drawdown = None  # 回撤

        self.show_withdraw = show_withdraw
        self.show_statistics = show_statistics
        self.max = copy.deepcopy(self.init_capital)

    def _statistics(self):
        raise NotImplemented

    def _reward(self, action):
        """这是股票的交易框架"""
        if action == -2:
            if self.direction == 1:  # 1).如有多头清空所有多头同时增加空头;
                self.direction = -1
                self.current_act_T = self.current_T
                self.current_act_price = self.Close[self.current_act_T]
                assert self.act_cost > 0
                r = self.num_act * (
                            self.fold * (self.current_act_price - self.last_act_price) - 2 * self.act_cost)  # 出多头再进空头
                self.total = self.total + r
                self.last_act_T = copy.deepcopy(self.current_act_T)
                self.last_act_price = copy.deepcopy(self.current_act_price)
                return r

            elif self.direction == 0:  # 2).如果没有多头,就增加空头"
                self.direction = -1
                self.current_act_T = self.current_T
                self.current_act_price = self.Close[self.current_act_T]
                assert self.act_cost > 0
                r = -self.num_act * self.act_cost
                self.total = self.total + r
                self.last_act_T = copy.deepcopy(self.current_act_T)
                self.last_act_price = copy.deepcopy(self.current_act_price)
                return r

            elif self.direction == -1:
                return 0


        elif action == -1:  # 卖出信号
            if self.direction == 1:  # 上次是持有多头
                self.direction = 0  # 卖出了，就没有场内头寸了，所以是0
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                assert self.act_cost > 0
                r = self.num_act * [
                    self.fold * (self.current_act_price - self.last_act_price) - self.act_cost]  # 当前收益=上一个价格-当前操作价格
                self.total = self.total + r  # 当前总资产
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                return r

            elif self.direction == 0:
                self.direction = -1
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                assert self.act_cost > 0
                r = -self.num_act * self.act_cost
                self.total = self.total + r
                return r

            elif self.direction == -1:
                return 0


        elif action == 0:  # 动作不变
            return 0


        elif action == 1:  # 动作买入
            if self.direction == 1:
                return 0

            elif self.direction == 0:  # 场内没有头寸
                self.direction = 1  # 买入头寸
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                r = -self.act_cost * self.num_act
                self.total = self.total + r
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                return r

            elif self.direction == -1:
                self.direction = 0
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                r = self.num_act * [self.fold * (self.last_act_price - self.current_act_price) - self.act_cost]
                self.total = self.total + r
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                return r

        elif action == 2:
            if self.direction == 1:
                return 0

            elif self.direction == 0:
                self.direction = 1
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                r = -self.num_act * self.act_cost
                self.total += r
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                return r

            elif self.direction == -1:
                self.direction = 1
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                r = self.num_act * (self.fold * (self.last_act_price - self.current_act_price) - 2 * self.act_cost)
                self.total += r
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                return r

    def _done(self):
        if self.total < self.Close[self.current_T] * 0.8:  # 资金不足以开一手
            self.close()
            return 1
        elif self.current_T == self.len_T:  # 达到索引中点
            return 1
        else:
            return 0

    def _info(self):
        if self.show_statistics == True:
            self._statistics()
        return ''

    def step(self, a):
        """Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        action = self._action_set[a]
        reward = self._reward(action)
        done = self._done()
        info = self._info()

        self.current_T += 1
        observations = self.data[:, self.current_T - self.windows + 1: self.current_T + 1]
        return observations, reward, done, info

    def reset(self):
        """
        :return: observation (object): agent's observation of the current environment
        """
        self.start_T = self.np_random.randint(self.warm_up, self.len_T)
        self.current_T = copy.deepcopy(self.start_T)
        observations = self.data[:, self.start_T - self.windows + 1: self.start_T + 1]
        return observations

    def render(self, mode='human'):
        raise NotImplemented

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class singal_fin_nodeath(Env):
    # 动作空间太多,容易坍缩至0,即不买也不卖.
    ACTION_MEANING = {
        -2: "大空,1).如有多头清空所有多头同时增加空头;2).如果没有多头,就增加空头",
        -1: "空 , 1).如有多头清空所有多头后不买入;   2).如果没有多头,就增加空头",
        0: "不变",
        1: "多,   1).如有空头清空所有空头后不买入;   2).如果没有空头,就增加多头",
        2: "大多, 1).如有空头清空所有空头同时增加多头;2).如果没有空头,就增加多头",
    }

    def __init__(self,
                 game='Futures',
                 code="RB",
                 mode=None,
                 data=None,
                 date=None,
                 seed=None,
                 data_shape=[],
                 windows=128,
                 warm_up=None,
                 init_capital=100000000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=3.4,
                 signal_index=None,
                 num_act=1,
                 fold=10,
                 scale=False,
                 ):
        """面向期货的,能双向交易盈利
        :param game:
        :param code:
        :param mode: life or nodeath
        :param data:
        :param date:
        :param seed:
        :param data_shape:
        :param windows:
        :param warm_up:
        :param init_capital:
        :param show_withdraw:
        :param show_statistics:
        :param cost:
        :param signal_index: 必须指定信号是第几行.
        :param num_act: 执行的数量
        :param fold:
        """
        self._action_set = [-2, -1, 0, 1, 2]  # 0:卖, 1:不变, 2:买
        self.action_space = spaces.Discrete(len(self._action_set))
        self.mode = "nodeath"
        self.scale = scale
        if data is None:
            raise ValueError
        self.data = data
        self.date = date
        self.Date = [d[:10] for d in self.date]
        self.Time = [d[11:] for d in self.date]
        self.signal = self.data[-1, :]  # 信号
        self.signal_inds = np.where(self.signal != 0)[0]  # 不为0的信号索引
        self.terminal = self.signal_inds[-1]

        self.Close = self.data[3, :]  # 不同数据可能所在的位置不一样
        if data_shape == []:
            (self.data_n, self.data_m) = self.data.shape
        self.windows = windows
        if warm_up is None:
            self.warm_up = round(self.data_m * 0.1)  # 预热
        else:
            self.warm_up = warm_up
        self.start_T = None  # reset()之后的开始时间索引
        self.current_T = None  # 当前的时间索引
        self.len_T = len(self.Time)
        assert self.len_T == self.data_m
        self.seed(seed)
        self.observation_space = spaces.Box(low=-10000, high=100000, dtype=np.uint8, shape=(self.data_n, self.windows))

        # financial affairs
        self.code = code
        self.init_capital = init_capital  # 初始资金
        self.fold = fold  # 每条代表的价格,如: 螺纹钢1跳为10元/手
        self.current_act_T = None  # 这次操作的时间索引
        self.current_act_price = None  # 这次操作价格
        self.returns = 0  # 收益
        self.cash = copy.deepcopy(self.init_capital)  # 场外资金(现金)
        self.value_in_market = 0  # 市场中的资金
        # self.total = 0  # 当前总资产
        self.last_act_T = None  # 上次操作的时间索引,(操作有买入/卖出)
        self.last_act_price = None  # 上次操作的价格
        self.act_cost = cost  # 每次操作的成本(对于股票来说卖出才有手续费)
        self.commission = 0
        self.num_act = num_act  # 每次操作的数量
        self.direction = 0  # 持有的头寸方向
        self.drawdown = None  # 回撤

        self.show_withdraw = show_withdraw
        self.show_statistics = show_statistics
        self.max = copy.deepcopy(self.init_capital)

    def _statistics(self):
        raise NotImplemented

    def _reward(self, action):
        """这是股票的交易框架"""
        if action == -2:
            if self.direction == 1:  # 1).如有多头清空所有多头同时增加空头;
                self.direction = -1
                self.current_act_T = self.current_T
                self.current_act_price = self.Close[self.current_act_T]
                assert self.act_cost > 0
                r = self.num_act * (
                            self.fold * (self.current_act_price - self.last_act_price) - 2 * self.act_cost)  # 出多头再进空头
                self.returns += r
                self.last_act_T = copy.deepcopy(self.current_act_T)
                self.last_act_price = copy.deepcopy(self.current_act_price)
                return r

            elif self.direction == 0:  # 2).如果没有多头,就增加空头"
                self.direction = -1
                self.current_act_T = self.current_T
                self.current_act_price = self.Close[self.current_act_T]
                assert self.act_cost > 0
                r = -self.num_act * self.act_cost
                self.returns += r
                self.last_act_T = copy.deepcopy(self.current_act_T)
                self.last_act_price = copy.deepcopy(self.current_act_price)
                return r

            elif self.direction == -1:
                return 0


        elif action == -1:  # 卖出信号
            if self.direction == 1:  # 上次是持有多头
                self.direction = 0  # 卖出了，就没有场内头寸了，所以是0
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                assert self.act_cost > 0
                r = self.num_act * (self.fold * (
                            self.current_act_price - self.last_act_price) - self.act_cost)  # 当前收益=上一个价格-当前操作价格
                self.returns += r
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                return r

            elif self.direction == 0:
                self.direction = -1
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                assert self.act_cost > 0
                r = -self.num_act * self.act_cost
                self.returns += r
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                return r

            elif self.direction == -1:
                return 0


        elif action == 0:  # 动作不变
            return 0


        elif action == 1:  # 动作买入
            if self.direction == 1:
                return 0

            elif self.direction == 0:  # 场内没有头寸
                self.direction = 1  # 买入头寸
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                r = -self.act_cost * self.num_act
                self.returns += r
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                return r

            elif self.direction == -1:
                self.direction = 0
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                r = self.num_act * (self.fold * (self.last_act_price - self.current_act_price) - self.act_cost)
                self.returns += r
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                return r

        elif action == 2:
            if self.direction == 1:
                return 0

            elif self.direction == 0:
                self.direction = 1
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                r = -self.num_act * self.act_cost
                self.returns += r
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                return r

            elif self.direction == -1:
                self.direction = 1
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                r = self.num_act * (self.fold * (self.last_act_price - self.current_act_price) - 2 * self.act_cost)
                self.returns += r
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                return r

    def _done(self):
        if self.current_T == self.terminal:  # 达到索引中点
            return 1
        else:
            return 0

    def _info(self):
        if self.show_statistics == True:
            self._statistics()
        return ''

    def _scale(self, observations):
        """TODO先测试在神经网络中的BN是否ok,即设置为(-100,100)"""
        pass

    def step(self, a):
        """Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        action = self._action_set[a]
        reward = self._reward(action)
        done = self._done()
        info = self._info()
        if done:  # 信号非零的走完了,就返回最后一个信号的下一个时刻的观察窗
            observations = self.data[:, self.current_T - self.windows + 1: self.current_T + 1]
            return observations, reward, done, info
        self.current_T = self.signal_Index.__next__()
        observations = self.data[:, self.current_T - self.windows + 1: self.current_T + 1]
        if self.scale == True:
            observations = self._scale(observations)
        return observations, reward, done, info

    def reset(self):
        """
        :return: observation (object): agent's observation of the current environment
        """
        self.signal_Index = self.signal_inds.__iter__()
        for _ in self.signal_Index:
            if _ > self.warm_up:
                ind = _  # 大于预热的 最小的 不为0的 信号的 索引
                break
        self.start_T = ind
        self.current_T = copy.deepcopy(self.start_T)
        observations = self.data[:, self.start_T - self.windows + 1: self.start_T + 1]
        if self.scale == True:
            observations = self._scale(observations)
        return observations

    def render(self, mode='human'):
        return self.Close[self.current_T - 4: self.current_T + 1], self.current_T

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class singal_fin_nodeathandstop(Env):
    ACTION_MEANING = {
        -1: "空 , 1).如有多头清空所有多头后不买入;   2).如果没有多头,就增加空头",
        1: "多,   1).如有空头清空所有空头后不买入;   2).如果没有空头,就增加多头",
    }

    def __init__(self,
                 game='Futures',
                 code="RB",
                 mode=None,
                 data=None,
                 date=None,
                 seed=None,
                 data_shape=[],
                 windows=128,
                 warm_up=None,
                 init_capital=100000000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=0.,
                 signal_index=None,
                 num_act=1,
                 fold=10,
                 scale=False,
                 ):
        """面向期货的,能双向交易盈利
        :param game:
        :param code:
        :param mode:
        :param data:
        :param date:
        :param seed:
        :param data_shape:
        :param windows:
        :param warm_up:
        :param init_capital:
        :param show_withdraw:
        :param show_statistics:
        :param cost:
        :param signal_index: 必须指定信号是第几行.
        :param num_act: 执行的数量
        :param fold:
        """
        self._action_set = [-1, 1]  # -1:卖, 1:买
        self.action_space = spaces.Discrete(len(self._action_set))
        self.mode = "nodeath"
        self.scale = scale
        if data is None:
            raise ValueError
        self.data = data
        self.date = date
        self.Date = [d[:10] for d in self.date]
        self.Time = [d[11:] for d in self.date]
        self.signal = self.data[-1, :]  # 信号
        self.signal_inds = np.where(self.signal != 0)[0]  # 不为0的信号索引
        self.terminal = self.signal_inds[-2]

        self.Close = self.data[3, :]  # 不同数据可能所在的位置不一样
        if data_shape == []:
            (self.data_n, self.data_m) = self.data.shape
        self.windows = windows
        if warm_up is None:
            self.warm_up = round(self.data_m * 0.1)  # 预热
        else:
            self.warm_up = warm_up
        self.start_T = None  # reset()之后的开始时间索引
        self.current_T = None  # 当前的时间索引
        self.len_T = len(self.Time)
        assert self.len_T == self.data_m
        self.seed(seed)
        self.observation_space = spaces.Box(low=-10000, high=100000, dtype=np.uint8, shape=(self.data_n, self.windows))

        # financial affairs
        self.code = code
        self.init_capital = init_capital  # 初始资金
        self.fold = fold  # 每条代表的价格,如: 螺纹钢1跳为10元/手
        self.current_act_T = None  # 这次操作的时间索引
        self.current_act_price = None  # 这次操作价格
        self.returns = 0  # 收益
        self.cash = copy.deepcopy(self.init_capital)  # 场外资金(现金)
        self.value_in_market = 0  # 市场中的资金
        # self.total = 0  # 当前总资产
        self.next_T = None  # 上次操作的时间索引,(操作有买入/卖出)
        self.next_act_price = None  # 上次操作的价格
        self.act_cost = cost  # 每次操作的成本(对于股票来说卖出才有手续费)
        self.commission = 0
        self.num_act = num_act  # 每次操作的数量
        self.direction = 0  # 持有的头寸方向
        self.drawdown = None  # 回撤

        self.show_withdraw = show_withdraw
        self.show_statistics = show_statistics
        self.max = copy.deepcopy(self.init_capital)

    def _statistics(self):
        raise NotImplemented

    def _reward(self, action):
        """这是股票的交易框架"""
        if action == -1:  # 卖出信号
            self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
            self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
            self.next_act_price = self.Close[self.next_T]

            r = self.num_act * (
                        self.fold * (self.current_act_price - self.next_act_price) - self.act_cost)  # 当前收益=上一个价格-当前操作价格
            self.returns += r
            return r

        elif action == 1:  # 动作买入
            self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
            self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
            self.next_act_price = self.Close[self.next_T]

            r = self.num_act * (self.fold * (self.next_act_price - self.current_act_price) - self.act_cost)
            self.returns += r
            return r

    def _done(self):
        if self.current_T == self.terminal:  # 达到索引中点
            return 1
        else:
            return 0

    def _info(self):
        if self.show_statistics == True:
            self._statistics()
        return ''

    def _scale(self, o):
        """TODO有些特征可能不能粗暴地归一"""
        means = np.mean(o, axis=1, keepdims=True)
        vars = np.var(o, axis=1, keepdims=True).squeeze()
        stds = np.array([np.sqrt(v) for v in vars]).reshape(self.data_n, 1)
        return (o - means) * 1000 / stds

    def step(self, a):
        """Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.next_T = self.signal_Index.__next__()
        action = self._action_set[a]
        reward = self._reward(action)
        done = self._done()
        info = self._info()
        self.current_T = self.next_T
        # if done:  # 信号非零的走完了,就返回最后一个信号的下一个时刻的观察窗
        #     observations = self.data[:, self.current_T - self.windows + 1: self.current_T + 1]
        #     return observations, reward, done, info

        observations = self.data[:, self.current_T - self.windows + 1: self.current_T + 1]
        if self.scale == True:
            observations = self._scale(observations)
        return observations, reward, done, info

    def reset(self):
        """
        :return: observation (object): agent's observation of the current environment
        """
        self.signal_Index = self.signal_inds.__iter__()
        for _ in self.signal_Index:
            if _ > self.warm_up:
                ind = _  # 大于预热的 最小的 不为0的 信号的 索引
                break
        self.start_T = ind
        self.current_T = copy.deepcopy(self.start_T)
        observations = self.data[:, self.start_T - self.windows + 1: self.start_T + 1]
        if self.scale == True:
            observations = self._scale(observations)
        return observations

    def render(self, mode='human'):
        return self.Close[self.current_T - 4: self.current_T + 1], self.current_T

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class fin_daily(Env):
    # 按天自由交易,不跨天
    pass


class singal_fin_daily(Env):
    # 按天及signal来交易,不跨天
    ACTION_MEANING = {
        -1: "空 , 1).如有多头清空所有多头后不买入;   2).如果没有多头,就增加空头",
        1: "多,   1).如有空头清空所有空头后不买入;   2).如果没有空头,就增加多头",
    }

    def __init__(self,
                 game='Futures',
                 code="RB",
                 mode=None,
                 data=None,
                 date=None,
                 seed=None,
                 data_shape=[],
                 windows=128,
                 warm_up=None,
                 init_capital=100000000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=0.,
                 signal_index=None,
                 num_act=1,
                 fold=10,
                 scale=False,
                 ):
        """面向期货的,能双向交易盈利
        :param game:
        :param code:
        :param mode:
        :param data:
        :param date:
        :param seed:
        :param data_shape:
        :param windows:
        :param warm_up:
        :param init_capital:
        :param show_withdraw:
        :param show_statistics:
        :param cost:
        :param signal_index: 必须指定信号是第几行.
        :param num_act: 执行的数量
        :param fold:
        """
        self._action_set = [-1, 1]  # -1:卖, 1:买
        self.action_space = spaces.Discrete(len(self._action_set))
        self.mode = "nodeath"
        self.scale = scale
        if data is None:
            raise ValueError
        self.data = data
        self.date = date
        self.Date = [d[:10] for d in self.date]
        self.Time = [d[11:] for d in self.date]
        self.signal = self.data[-1, :]  # 信号
        self.signal_inds = np.where(self.signal != 0)[0]  # 不为0的信号索引
        self.terminal = self.signal_inds[-2]

        self.Close = self.data[3, :]  # 不同数据可能所在的位置不一样
        if data_shape == []:
            (self.data_n, self.data_m) = self.data.shape
        self.windows = windows
        if warm_up is None:
            self.warm_up = round(self.data_m * 0.1)  # 预热
        else:
            self.warm_up = warm_up
        self.start_T = None  # reset()之后的开始时间索引
        self.current_T = None  # 当前的时间索引
        self.len_T = len(self.Time)
        assert self.len_T == self.data_m
        self.seed(seed)
        self.observation_space = spaces.Box(low=-10000, high=100000, dtype=np.uint8, shape=(self.data_n, self.windows))

        # financial affairs
        self.code = code
        self.init_capital = init_capital  # 初始资金
        self.fold = fold  # 每条代表的价格,如: 螺纹钢1跳为10元/手
        self.current_act_T = None  # 这次操作的时间索引
        self.current_act_price = None  # 这次操作价格
        self.returns = 0  # 收益
        self.cash = copy.deepcopy(self.init_capital)  # 场外资金(现金)
        self.value_in_market = 0  # 市场中的资金
        # self.total = 0  # 当前总资产
        self.next_T = None  # 上次操作的时间索引,(操作有买入/卖出)
        self.next_act_price = None  # 上次操作的价格
        self.act_cost = cost  # 每次操作的成本(对于股票来说卖出才有手续费)
        self.commission = 0
        self.num_act = num_act  # 每次操作的数量
        self.direction = 0  # 持有的头寸方向
        self.drawdown = None  # 回撤

        self.show_withdraw = show_withdraw
        self.show_statistics = show_statistics
        self.max = copy.deepcopy(self.init_capital)

    def _statistics(self):
        raise NotImplemented

    def _reward(self, action):
        """这是股票的交易框架"""
        if action == -1:  # 卖出信号
            self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
            self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
            self.next_act_price = self.Close[self.next_T]

            r = self.num_act * (
                        self.fold * (self.current_act_price - self.next_act_price) - self.act_cost)  # 当前收益=上一个价格-当前操作价格
            self.returns += r
            return r

        elif action == 1:  # 动作买入
            self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
            self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
            self.next_act_price = self.Close[self.next_T]

            r = self.num_act * (self.fold * (self.next_act_price - self.current_act_price) - self.act_cost)
            self.returns += r
            return r

    def _done(self):
        if self.current_T == self.terminal:  # 达到索引中点
            return 1
        else:
            return 0

    def _info(self):
        if self.show_statistics == True:
            self._statistics()
        return ''

    def _scale(self, o):
        """TODO有些特征可能不能粗暴地归一"""
        means = np.mean(o, axis=1, keepdims=True)
        vars = np.var(o, axis=1, keepdims=True).squeeze()
        stds = np.array([np.sqrt(v) for v in vars]).reshape(self.data_n, 1)
        return (o - means) * 1000 / stds

    def step(self, a):
        """Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.next_T = self.signal_Index.__next__()
        action = self._action_set[a]
        reward = self._reward(action)
        done = self._done()
        info = self._info()
        self.current_T = self.next_T
        # if done:  # 信号非零的走完了,就返回最后一个信号的下一个时刻的观察窗
        #     observations = self.data[:, self.current_T - self.windows + 1: self.current_T + 1]
        #     return observations, reward, done, info

        observations = self.data[:, self.current_T - self.windows + 1: self.current_T + 1]
        if self.scale == True:
            observations = self._scale(observations)
        return observations, reward, done, info

    def reset(self):
        """
        :return: observation (object): agent's observation of the current environment
        """
        self.signal_Index = self.signal_inds.__iter__()
        for _ in self.signal_Index:
            if _ > self.warm_up:
                ind = _  # 大于预热的 最小的 不为0的 信号的 索引
                break
        self.start_T = ind
        self.current_T = copy.deepcopy(self.start_T)
        observations = self.data[:, self.start_T - self.windows + 1: self.start_T + 1]
        if self.scale == True:
            observations = self._scale(observations)
        return observations

    def render(self, mode='human'):
        return self.Close[self.current_T - 4: self.current_T + 1], self.current_T

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class assembly_fin(Env):
    def __init__(self, data_path='../data/',
                 game='Futures',
                 mode="nodeath",
                 seed=123,
                 windows=60,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=3.4,
                 signal_index=-1,
                 fold=10,
                 **kwargs):
        self.seed(seed)
        self.envs = []
        self.mode = mode
        for file in os.listdir(data_path):
            if file[-3:] == 'npy':
                temp = np.load(os.path.join(data_path, file), allow_pickle=True)

                code = file[:6]
                data = temp[1:, :]
                date = temp[0]
                warm_up = round(data.shape[1] * 0.1)
                if self.mode == "nodeath":
                    sf = singal_fin_nodeath(
                        game=game,
                        code=code,
                        mode=mode,
                        data=data,
                        date=date,
                        seed=seed,
                        windows=windows,
                        warm_up=warm_up,
                        show_withdraw=show_withdraw,
                        show_statistics=show_statistics,
                        cost=cost,
                        signal_index=signal_index,
                        fold=fold,
                        scale=scale)
                elif mode == "nodeathandstop":
                    sf = singal_fin_nodeathandstop(
                        game=game,
                        code=code,
                        mode=mode,
                        data=data,
                        date=date,
                        seed=seed,
                        windows=windows,
                        warm_up=warm_up,
                        show_withdraw=show_withdraw,
                        show_statistics=show_statistics,
                        cost=cost,
                        signal_index=signal_index,
                        fold=fold,
                        scale=scale)
                self.envs.append(sf)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        _id = self.np_random.randint(len(self.envs))
        self.current_env = self.envs[_id]
        o = self.current_env.reset()
        return o

    def step(self, a):
        observations, reward, done, info = self.current_env.step(a)
        return observations, reward, done, info

    def render(self, mode='human'):
        self.current_env.render()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class continus_Fin_Futures_holding_reward(Env):
    ACTION_MEANING = {
        # -2: "大空,1).如有多头清空所有多头同时增加空头;2).如果没有多头,就增加空头",
        -1: "空 , 1).如有多头清空所有多头后不买入;   2).如果没有多头,就增加空头",
        1: "多,   1).如有空头清空所有空头后不买入;   2).如果没有空头,就增加多头",
        # 2: "大多, 1).如有空头清空所有空头同时增加多头;2).如果没有空头,就增加多头",
    }

    def __init__(self,
                 code=None,
                 data=None,  # 必须是单独一个合约的数据,非连续主力合约数据
                 date=None,
                 Close=None,
                 seed=None,
                 data_shape=[],
                 windows=128,
                 warm_up=None,
                 init_capital=10000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=-3.4,
                 fold=10,  # 杠杆
                 ):
        """期货交易,双向盈利,持续交易,持有时有反馈"""
        self._action_set = [-1, 1]  #
        self.action_space = spaces.Discrete(len(self._action_set))
        if data is None:
            raise ValueError
        self.code = code
        self.data = data
        self.date = date
        self.Date = [d[:10] for d in self.date]
        self.Time = [d[11:] for d in self.date]
        if Close is None:
            self.Close = self.data[3, :]
        else:
            self.Close = Close
        if data_shape == []:
            (self.data_n, self.data_m) = self.data.shape
        self.windows = windows

        self.warm_up = warm_up if warm_up else round(data.shape[1] * 0.1)  # 预热

        self.len_T = len(self.Time)
        self.seed(seed)
        self.observation_space = spaces.Box(low=-1000, high=10000, dtype=np.uint8, shape=(self.data_n, self.windows))
        self.fold = fold
        self.show_withdraw = show_withdraw
        self.show_statistics = show_statistics
        self.statistics = []

        # financial affairs

        self.start_T = None  # reset()之后的开始时间索引
        self.current_T = None  # 当前的时间索引
        self.init_capital = init_capital  # 初始资金
        self.act_cost = cost  # 每次操作的成本(对于股票来说卖出才有手续费)
        assert self.act_cost < 0
        self._initialise_capital()

    def _initialise_capital(self):
        self.current_act_T = None  # 这次操作的时间索引
        self.current_act_price = None  # 这次操作价格
        self.total = copy.deepcopy(self.init_capital)  # 当前总资产
        self.last_act_T = None  # 上次操作的时间索引,(操作有买入/卖出)
        self.last_act_price = None  # 上次操作的价格
        self.commission = 0
        self.num_act = 1  # 每次操作的数量
        self.direction = 0  # 持有的头寸方向
        self.drawdown = None  # 回撤
        self.reward = 0
        self.max = copy.deepcopy(self.init_capital)

    def _reward(self, action):
        """这是股票的交易框架"""
        if action == -1:  # 动作卖出
            if self.direction == 1:
                self.reward = 2 * self.act_cost + (
                            self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                self.total += self.reward
                self.direction = -1  # 这次操作价格

            elif self.direction == 0:
                self.reward = self.act_cost  # 收益
                self.total += self.reward  # 当前总资产
                self.direction = -1

            elif self.direction == -1:
                self.reward = self.direction * (self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                self.total += self.reward

        elif action == 1:  # 动作买入
            if self.direction == 1:
                self.reward = (self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                self.total += self.reward

            elif self.direction == 0:
                self.reward = self.act_cost  # 收益
                self.total += self.reward
                self.direction = 1

            elif self.direction == -1:
                self.reward = 2 * self.act_cost + self.direction * (
                            self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                self.total += self.reward
                self.direction = 1

        return self.reward

    def _done(self):
        if self.total < self.Close[self.current_T] * 10 * 0.08:  # 资金不足
            self.close()
            return 1
        elif self.current_T == self.len_T-1:  # 达到索引中点
            return 1
        else:
            return 0

    def _info(self):
        if self.show_statistics == True:
            return self.statistics[-1]  # 返回最后一个,即当前的统计信息

    def _statistics(self):  # 只能通过self.reset() 和self.step()调用
        # self.show_statistics == True才调用
        li = {}
        li['code'] = self.code
        li['DateTime'] = self.date[self.current_T]
        li['rewards'] = self.reward
        li['current_T'] = self.current_T
        li['current_price'] = self.Close[self.current_T]
        li['current_position'] = self.direction
        li['total'] = self.total
        self.statistics.append(li)

    def step(self, a):
        action = self._action_set[a]
        reward = self._reward(action)

        if self.show_statistics:
            self._statistics()
        self.current_T += 1
        done = self._done()
        info = self._info()

        observations = self.data[:, self.current_T - self.windows + 1: self.current_T + 1]
        return observations.astype('float64'), reward, done, info

    def reset(self, isRandomStart=False):
        """
        isRandomStart: 从环境的开始走到结尾/随机开始
        :return: observation (object): agent's observation of the current environment
        """
        self._initialise_capital()
        self.start_T = self.np_random.randint(self.warm_up, self.len_T) if isRandomStart else self.warm_up
        self.current_T = copy.deepcopy(self.start_T)
        observations = self.data[:, self.start_T - self.windows + 1: self.start_T + 1]
        if self.show_statistics:
            self.statistics = []  # 每次重置都清空统计信息
            self._statistics()  # 记录初始的时间,价格以及总资产
        return observations.astype('float64')

    def render(self, mode='human'):
        raise NotImplemented

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class continus_Fin_Futures_holding_reward_pic(Env):
    ACTION_MEANING = {
        # -2: "大空,1).如有多头清空所有多头同时增加空头;2).如果没有多头,就增加空头",
        -1: "空 , 1).如有多头清空所有多头后不买入;   2).如果没有多头,就增加空头",
        1: "多,   1).如有空头清空所有空头后不买入;   2).如果没有空头,就增加多头",
        # 2: "大多, 1).如有空头清空所有空头同时增加多头;2).如果没有空头,就增加多头",
    }

    def __init__(self,
                 code=None,
                 data=None,  # 必须是单独一个合约的数据,非连续主力合约数据
                 date=None,
                 Close=None,
                 seed=None,
                 data_shape=[],
                 windows=60,
                 warm_up=None,
                 init_capital=10000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=-3.4,
                 fold=10,  # 杠杆
                 drawdown=0.2,
                 ):
        """期货交易,双向盈利,持续交易,持有时有反馈"""
        self._action_set = [-1, 1]  #
        self.action_space = spaces.Discrete(len(self._action_set))
        if data is None:
            raise ValueError
        self.code = code
        self.data = data
        self.date = date
        self.Date = [d[:10] for d in self.date]
        self.Time = [d[11:] for d in self.date]
        if Close is None:
            self.Close = self.data[3, :]
        else:
            self.Close = Close


        self.windows = windows

        self.warm_up = warm_up if warm_up else round(data.shape[1] * 0.1)  # 预热

        self.len_T = len(self.Time)
        self.seed(seed)
        self.observation_space = spaces.Box(low=-1000, high=10000, dtype=np.uint8, shape=(self.windows*3, self.windows*3))
        self.fold = fold
        self.show_withdraw = show_withdraw
        self.show_statistics = show_statistics
        self.statistics = []

        # financial affairs
        self.drawdown = drawdown  # 回撤
        self.start_T = None  # reset()之后的开始时间索引
        self.current_T = None  # 当前的时间索引
        self.init_capital = init_capital  # 初始资金
        self.act_cost = cost  # 每次操作的成本(对于股票来说卖出才有手续费)
        assert self.act_cost < 0
        self._initialise_capital()

    def _initialise_capital(self, total=None, max=None):
        self.iters = 0
        self.current_act_T = None  # 这次操作的时间索引
        self.current_act_price = None  # 这次操作价格
        self.total = total if total else copy.deepcopy(self.init_capital)  # 如果给定新的总资产, 否则默认总资产
        self.last_act_T = None  # 上次操作的时间索引,(操作有买入/卖出)
        self.last_act_price = None  # 上次操作的价格
        self.commission = 0
        self.num_act = 1  # 每次操作的数量
        self.direction = 0  # 持有的头寸方向

        self.reward = 0
        self.max = max if max else copy.deepcopy(self.init_capital)

    def _reward(self, action):
        """这是股票的交易框架"""
        if action == -1:  # 动作卖出
            if self.direction == 1:
                self.reward = 2 * self.act_cost + (
                            self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                self.total += self.reward
                self.direction = -1  # 这次操作价格

            elif self.direction == 0:
                self.reward = self.act_cost  # 收益
                self.total += self.reward  # 当前总资产
                self.direction = -1

            elif self.direction == -1:
                self.reward = self.direction * (self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                self.total += self.reward

        elif action == 1:  # 动作买入
            if self.direction == 1:
                self.reward = (self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                self.total += self.reward

            elif self.direction == 0:
                self.reward = self.act_cost  # 收益
                self.total += self.reward
                self.direction = 1

            elif self.direction == -1:
                self.reward = 2 * self.act_cost + self.direction * (
                            self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                self.total += self.reward
                self.direction = 1

        self.max = self.total if self.max < self.total else self.max
        return self.reward

    def _done(self):
        if self.total < self.Close[self.current_T] * 10 * 0.08:  # 资金不足

            return 1
        elif self.current_T == self.len_T-1:  # 达到索引终点
            return 2
        elif self.max/self.total > 1 + self.drawdown:  #回撤超过回撤率限制
            return 3
        else:
            return 0

    def _info(self):
        if self.show_statistics == True:
            return self.statistics[-1]  # 返回最后一个,即当前的统计信息

    def _statistics(self):  # 只能通过self.reset() 和self.step()调用
        # self.show_statistics == True才调用
        li = {}
        li['code'] = self.code
        li['DateTime'] = self.date[self.current_T]
        li['rewards'] = self.reward
        li['current_T'] = self.current_T
        li['current_price'] = self.Close[self.current_T]
        li['current_position'] = self.direction
        li['total'] = self.total
        self.statistics.append(li)

    def step(self, a):
        action = self._action_set[a]
        reward = self._reward(action)

        if self.show_statistics:
            self._statistics()
        self.current_T += 1
        done = self._done()
        info = self._info()

        _observations = self.data[:, self.current_T - self.windows + 1: self.current_T + 1]
        self.iters += 1
        self.observations = self.generate_pic(_observations).astype('float32')

        return self.observations, reward, done, info

    def generate_pic(self, _observations):
        _observations = _observations.astype('int')
        length = self.windows*3
        raw_pic = np.zeros((length, length))  # 空白图
        distance = _observations.max() - _observations.min()  # 判断是否超出图的范围
        # 以最高点为基准
        bench_index = _observations.max()
        obs = bench_index - _observations
        if distance < self.windows*3:
            for i in reversed(range(self.windows)):
                raw_pic[obs[3, i], (i+1)*3 - 1] = 1  # close
                raw_pic[obs[1, i]: obs[2, i], (i+1)*3 - 2] = 1  # high~low
                raw_pic[obs[0, i], (i+1)*3 - 3] = 1  # open
        else:
            for i in reversed(range(self.windows)):
                close_index = min(obs[3, i], length - 1)  # 不能超过图的边界
                open_index = min(obs[0, i], length - 1)  # 不能超过图的边界
                high_index = min(obs[1, i], length - 1)  # 不能超过图的边界
                low_index = min(obs[2, i], length - 1)  # 不能超过图的边界

                raw_pic[close_index, (i+1)*3 - 1] = 1  # close
                raw_pic[high_index: low_index, (i+1)*3 - 2] = 1  # high~low
                raw_pic[open_index, (i+1)*3 - 3] = 1  # open
        return raw_pic

    def reset(self, isRandomStart=False, total=None, max=None):
        """
        isRandomStart: 从环境的开始走到结尾/随机开始
        :return: observation (object): agent's observation of the current environment
        """
        self._initialise_capital(total, max)
        self.start_T = self.np_random.randint(self.warm_up, self.len_T) if isRandomStart else self.warm_up
        self.current_T = copy.deepcopy(self.start_T)
        _observations = self.data[:, self.start_T - self.windows + 1: self.start_T + 1]
        self.observations = self.generate_pic(_observations).astype('float32')
        if self.show_statistics:
            self.statistics = []  # 每次重置都清空统计信息
            self._statistics()  # 记录初始的时间,价格以及总资产
        return self.observations

    def render(self, mode='human'):
        Image.fromarray(self.observations*255).show()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class continus_Daily_Fin_Futures_holding_reward_pic(continus_Fin_Futures_holding_reward_pic):

    def __init__(self,
                 code=None,
                 data=None,  # 必须是单独一个合约的数据,非连续主力合约数据
                 date=None,
                 Close=None,
                 seed=None,
                 data_shape=[],
                 windows=60,
                 warm_up=None,
                 init_capital=10000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=-3.4,
                 fold=10,  # 杠杆
                 start_Time='09:30:00',
                 **kwargs,
                 ):
        super(continus_Daily_Fin_Futures_holding_reward_pic, self).__init__(
            code=code,
            data=data,  # 必须是单独一个合约的数据,非连续主力合约数据
            date=date,
            Close=Close,
            seed=seed,
            data_shape=data_shape,
            windows=windows,
            warm_up=warm_up,
            init_capital=init_capital,
            show_withdraw=show_withdraw,
            show_statistics=show_statistics,
            cost=cost,
            fold=fold,  # 杠杆
            **kwargs,
            )
        self.start_Time = start_Time
        self.setDate = set(self.Date)  # 按顺序整理出所有日期
        self.date_sorted = sorted(list(self.setDate))  # 按时间顺序排列日期
        self.warm_up = round(self.setDate.__len__() * 0.1)  # 当天开盘n分钟后开始交易
        self.checked_date = []

    def join_date_time(self, date='', time=''):
        return date + ' ' + time

    def _reward(self, action):
        """这是股票的交易框架"""
        if ((self.Date[self.current_T] < self.Date[self.current_T + 1]) and self.Time[
            self.current_T + 1] >= '09:00:00') or (self.Time[self.current_T] < '02:00:00' and
                                                   self.Time[
                                                       self.current_T + 1] > '09:00:00') or self.current_T + 2 == self.len_T:  # 当天收盘
            self.reward = self.act_cost + self.direction * (
                    self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
            self.total += self.reward
            self.direction = 0

        else:
            if action == -1:  # 动作卖出
                if self.direction == 1:
                    self.reward = 2 * self.act_cost + (
                            self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                    self.total += self.reward
                    self.direction = -1  # 这次操作价格

                elif self.direction == 0:
                    self.reward = self.act_cost  # 收益
                    self.total += self.reward  # 当前总资产
                    self.direction = -1

                elif self.direction == -1:
                    self.reward = self.direction * (
                            self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                    self.total += self.reward

            elif action == 1:  # 动作买入
                if self.direction == 1:
                    self.reward = (self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                    self.total += self.reward

                elif self.direction == 0:
                    self.reward = self.act_cost  # 收益
                    self.total += self.reward
                    self.direction = 1

                elif self.direction == -1:
                    self.reward = 2 * self.act_cost + self.direction * (
                            self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                    self.total += self.reward
                    self.direction = 1
        self.max = self.total if self.max < self.total else self.max
        return self.reward

    def step(self, a):
        action = self._action_set[a]
        reward = self._reward(action)

        if self.show_statistics:
            self._statistics()

        if ((self.Date[self.current_T] < self.Date[self.current_T + 1]) and self.Time[
            self.current_T + 1] >= '09:00:00') or (self.Time[self.current_T] < '02:00:00' and
                                                   self.Time[self.current_T + 1] > '09:00:00'):  # 当天收盘
            self.next_date = self.join_date_time(self.Date[self.current_T + 1], self.start_Time)  # 跳到下一天的开始交易时间
            self.current_T = np.where(self.date == self.next_date)[0].item()
            self.checked_date.append(self.next_date)
        else:
            self.current_T += 1
        info = self._info()
        done = self._done()

        self.iters += 1
        _observations = self.data[:, self.current_T - self.windows + 1: self.current_T + 1]
        self.observations = self.generate_pic(_observations).astype('float32')
        return self.observations, reward, done, info

    def reset(self, isRandomStart=False, total=None, max=None):
        self._initialise_capital(total, max)

        self.start_date = self.date_sorted[
            self.np_random.randint(self.warm_up, len(self.date_sorted))] if isRandomStart else \
            self.date_sorted[self.warm_up]  # 开始日期,

        self.start_datetime = self.join_date_time(self.start_date, self.start_Time)

        self.start_T = np.where(self.date == self.start_datetime)[0].item()
        self.current_T = copy.deepcopy(self.start_T)
        _observations = self.data[:, self.start_T - self.windows + 1: self.start_T + 1]
        self.observations = self.generate_pic(_observations).astype('float32')
        if self.show_statistics:
            self.statistics = []  # 每次重置都清空统计信息
            self._statistics()  # 记录初始的时间,价格以及总资产
        return self.observations

class continus_Fin_Futures(continus_Fin_Futures_holding_reward):
    ACTION_MEANING = {
        # -2: "大空,1).如有多头清空所有多头同时增加空头;2).如果没有多头,就增加空头",
        -1: "空 , 1).如有多头清空所有多头后不买入;   2).如果没有多头,就增加空头",
        1: "多,   1).如有空头清空所有空头后不买入;   2).如果没有空头,就增加多头",
        # 2: "大多, 1).如有空头清空所有空头同时增加多头;2).如果没有空头,就增加多头",
    }

    def __init__(self,
                 code=None,
                 data=None,  # 必须是单独一个合约的数据,非连续主力合约数据
                 date=None,
                 Close=None,
                 seed=None,
                 data_shape=[],
                 windows=128,
                 warm_up=None,
                 init_capital=10000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=-3.4,
                 fold=10,  # 杠杆
                 ):
        """期货交易,双向盈利,持续交易,持有时无反馈"""
        super(continus_Fin_Futures, self).__init__(
            code=code,
            data=data,  # 必须是单独一个合约的数据,非连续主力合约数据
            date=date,
            Close=Close,
            seed=seed,
            data_shape=data_shape,
            windows=windows,
            warm_up=warm_up,
            init_capital=init_capital,
            show_withdraw=show_withdraw,
            show_statistics=show_statistics,
            cost=cost,
            fold=fold,  # 杠杆
        )

    def _reward(self, action):
        if action == -1:  # 动作卖出
            if self.direction == 1:  # 上次是已经买入了的
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                assert self.act_cost < 0
                self.reward = 2 * self.act_cost + (self.current_act_price - self.last_act_price) * self.fold  # 收益
                self.total += self.reward
                self.last_act_T = self.current_act_T  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = self.current_act_price  # 上次操作的价格
                self.direction = -1  # 卖出

            elif self.direction == 0:
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                assert self.act_cost < 0
                self.reward = self.act_cost
                self.total += self.reward
                self.last_act_T = self.current_act_T  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = self.current_act_price  # 上次操作的价格
                self.direction = -1

            elif self.direction == -1:
                self.reward = 0

        elif action == 1:  # 动作买入
            if self.direction == 0:  # 场内没有头寸
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                self.reward = self.act_cost
                self.total += self.reward
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                self.direction = 1  # 买入头寸

            elif self.direction == 1:
                self.reward = 0

            elif self.direction == -1:
                self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                self.reward = 2 * self.act_cost + (self.last_act_price - self.current_act_price) * self.fold
                self.total += self.reward
                self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                self.direction = 1

        return self.reward


class continus_Daily_Fin_Futures_holding_reward(continus_Fin_Futures_holding_reward):
    ACTION_MEANING = {
        # -2: "大空,1).如有多头清空所有多头同时增加空头;2).如果没有多头,就增加空头",
        -1: "空 , 1).如有多头清空所有多头后不买入;   2).如果没有多头,就增加空头",
        1: "多,   1).如有空头清空所有空头后不买入;   2).如果没有空头,就增加多头",
        # 2: "大多, 1).如有空头清空所有空头同时增加多头;2).如果没有空头,就增加多头",
    }

    def __init__(self,
                 code=None,
                 data=None,  # 必须是单独一个合约的数据,非连续主力合约数据
                 date=None,
                 Close=None,
                 seed=None,
                 data_shape=[],
                 windows=128,
                 warm_up=None,
                 init_capital=10000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=-3.4,
                 fold=10,  # 杠杆
                 start_Time='09:30:00',
                 ):
        """期货交易,双向盈利,日内交易,持有时有反馈"""
        super(continus_Daily_Fin_Futures_holding_reward, self).__init__(
            code=code,
            data=data,  # 必须是单独一个合约的数据,非连续主力合约数据
            date=date,
            Close=Close,
            seed=seed,
            data_shape=data_shape,
            windows=windows,
            warm_up=warm_up,
            init_capital=init_capital,
            show_withdraw=show_withdraw,
            show_statistics=show_statistics,
            cost=cost,
            fold=fold,  # 杠杆
        )
        self.start_Time = start_Time
        self.setDate = set(self.Date)  # 按顺序整理出所有日期
        self.date_sorted = sorted(list(self.setDate))  # 按时间顺序排列日期
        self.warm_up = round(self.setDate.__len__() * 0.1)  # 当天开盘n分钟后开始交易
        # self.Time_sorted = sorted(list(set(self.Time)))
        self.checked_date = []

    def join_date_time(self, date='', time=''):
        return date + ' ' + time

    def _reward(self, action):
        """这是股票的交易框架"""
        if ((self.Date[self.current_T] < self.Date[self.current_T + 1]) and self.Time[
            self.current_T + 1] >= '09:00:00') or (self.Time[self.current_T]<'02:00:00' and
            self.Time[self.current_T + 1]> '09:00:00') or self.current_T+2 == self.len_T:  # 当天收盘
            self.reward = self.act_cost + self.direction * (
                        self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
            self.total += self.reward
            self.direction = 0

        else:
            if action == -1:  # 动作卖出
                if self.direction == 1:
                    self.reward = 2 * self.act_cost + (
                                self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                    self.total += self.reward
                    self.direction = -1  # 这次操作价格

                elif self.direction == 0:
                    self.reward = self.act_cost  # 收益
                    self.total += self.reward  # 当前总资产
                    self.direction = -1

                elif self.direction == -1:
                    self.reward = self.direction * (
                                self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                    self.total += self.reward

            elif action == 1:  # 动作买入
                if self.direction == 1:
                    self.reward = (self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                    self.total += self.reward

                elif self.direction == 0:
                    self.reward = self.act_cost  # 收益
                    self.total += self.reward
                    self.direction = 1

                elif self.direction == -1:
                    self.reward = 2 * self.act_cost + self.direction * (
                                self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                    self.total += self.reward
                    self.direction = 1

        return self.reward

    def step(self, a):
        action = self._action_set[a]
        reward = self._reward(action)

        if self.show_statistics:
            self._statistics()

        if ((self.Date[self.current_T] < self.Date[self.current_T + 1]) and self.Time[
            self.current_T + 1] >= '09:00:00') or (self.Time[self.current_T] < '02:00:00' and
                                                   self.Time[self.current_T + 1] > '09:00:00'):  # 当天收盘
            self.next_date = self.join_date_time(self.Date[self.current_T + 1], self.start_Time)  # 跳到下一天的开始交易时间
            self.current_T = np.where(self.date == self.next_date)[0].item()
            self.checked_date.append(self.next_date)
        else:
            self.current_T += 1
        info = self._info()
        done = self._done()

        observations = self.data[:, self.current_T - self.windows + 1: self.current_T + 1]
        return observations.astype('float64'), reward, done, info

    def reset(self, isRandomStart=False):
        self._initialise_capital()

        self.start_date = self.date_sorted[self.np_random.randint(self.warm_up, len(self.date_sorted))] if isRandomStart else \
        self.date_sorted[self.warm_up]  # 开始日期,

        self.start_datetime = self.join_date_time(self.start_date, self.start_Time)

        self.start_T = np.where(self.date == self.start_datetime)[0].item()
        self.current_T = copy.deepcopy(self.start_T)
        observations = self.data[:, self.start_T - self.windows + 1: self.start_T + 1]
        if self.show_statistics:
            self.statistics = []  # 每次重置都清空统计信息
            self._statistics()  # 记录初始的时间,价格以及总资产
        return observations.astype('float64')


class continus_Daily_Fin_Futures(continus_Daily_Fin_Futures_holding_reward):
    def __init__(self,
                 code=None,
                 data=None,  # 必须是单独一个合约的数据,非连续主力合约数据
                 date=None,
                 Close=None,
                 seed=None,
                 data_shape=[],
                 windows=128,
                 warm_up=None,
                 init_capital=10000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=-3.4,
                 fold=10,  # 杠杆
                 start_Time='09:30:00',
                 ):
        """期货交易,双向盈利,日内交易,持有时无反馈"""
        super(continus_Daily_Fin_Futures, self).__init__(
            code=code,
            data=data,  # 必须是单独一个合约的数据,非连续主力合约数据
            date=date,
            Close=Close,
            seed=seed,
            data_shape=data_shape,
            windows=windows,
            warm_up=warm_up,
            init_capital=init_capital,
            show_withdraw=show_withdraw,
            show_statistics=show_statistics,
            cost=cost,
            fold=fold,  # 杠杆
        )
    def _reward(self, action):
        if ((self.Date[self.current_T] < self.Date[self.current_T + 1]) and self.Time[
            self.current_T + 1] >= '09:00:00') or (self.Time[self.current_T]<'02:00:00' and
            self.Time[self.current_T + 1]> '09:00:00') or self.current_T+2 == self.len_T:  # 当天收盘或者到数据结束

            self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
            self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
            self.reward = self.act_cost + self.direction * (self.current_act_price - self.last_act_price) * self.fold
            self.total += self.reward
            self.direction = 0

        else:
            if action == -1:  # 动作卖出
                if self.direction == 1:  # 上次是已经买入了的
                    self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                    self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                    self.reward = 2 * self.act_cost + (self.current_act_price - self.last_act_price) * self.fold  # 收益
                    self.total += self.reward
                    self.last_act_T = self.current_act_T  # 上次操作的时间索引,(操作有买入/卖出)
                    self.last_act_price = self.current_act_price  # 上次操作的价格
                    self.direction = -1  # 卖出

                elif self.direction == 0:
                    self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                    self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                    self.reward = self.act_cost
                    self.total += self.reward
                    self.last_act_T = self.current_act_T  # 上次操作的时间索引,(操作有买入/卖出)
                    self.last_act_price = self.current_act_price  # 上次操作的价格
                    self.direction = -1

                elif self.direction == -1:
                    self.reward = 0

            elif action == 1:  # 动作买入
                if self.direction == 0:  # 场内没有头寸
                    self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                    self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                    self.reward = self.act_cost
                    self.total += self.reward
                    self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                    self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                    self.direction = 1  # 买入头寸

                elif self.direction == 1:
                    self.reward = 0

                elif self.direction == -1:
                    self.current_act_T = self.current_T  # 当前操作索引=当前时间索引
                    self.current_act_price = self.Close[self.current_act_T]  # 当前操作价格=当前收盘价
                    self.reward = 2 * self.act_cost + (self.last_act_price - self.current_act_price) * self.fold
                    self.total += self.reward
                    self.last_act_T = copy.deepcopy(self.current_act_T)  # 上次操作的时间索引,(操作有买入/卖出)
                    self.last_act_price = copy.deepcopy(self.current_act_price)  # 上次操作的价格
                    self.direction = 1

        return self.reward

class Assembly_Fin(Env):
    def __init__(self, data_path='../data/',  # 文件存放路径
                 game=None,
                 seed=123,
                 windows=30,
                 **kwargs):
        self._seed = seed
        self.seed(self._seed)  # 随机选择环境的seed和每个环境中的seed不一样
        self.envs = []
        self.windows = windows
        self.data_path = data_path
        self.game = game
        self.kwargs = kwargs
        self.__generate()

    def __generate(self):
        for file in os.listdir(self.data_path):
            if file[-3:] == 'npy':
                temp = np.load(os.path.join(self.data_path, file), allow_pickle=True)
                code = file[:6]
                close = temp[4, :]  # 默认featuresmap的第四行为
                date = temp[0]
                data = temp[5:, :]
                sf = self.game(code=code, data=data, date=date, Close=close, windows=self.windows, seed=self._seed, **self.kwargs)
                self.envs.append(sf)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def _info(self):
        return self.current_env._info()

    def reset(self, isRandomStart=False, total=None):
        _id = self.np_random.randint(len(self.envs))
        self.current_env = self.envs[_id]
        o = self.current_env.reset(isRandomStart, total)
        return o

    def step(self, a):
        observations, reward, done, info = self.current_env.step(a)
        return observations, reward, done, info

    def render(self, mode='human'):
        self.current_env.render()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class Assembly_Fin_for_pic(Assembly_Fin):
    def __init__(self, data_path='../data/',  # 文件存放路径
                 game=None,
                 seed=123,
                 windows=60,
                 show_statistics=True,
                 init_capital=1000000,
                 **kwargs):
        self._seed = seed
        self.seed(self._seed)  # 随机选择环境的seed和每个环境中的seed不一样
        self.envs = []
        self.windows = windows
        self.data_path = data_path
        self.game = game
        self.show_statistics = show_statistics
        self.init_capital = init_capital
        self.kwargs = kwargs

        self.statistics = []  # 记录总的统计信息, 非done==1时充值不清0
        self.total = copy.deepcopy(self.init_capital)
        self.max = copy.deepcopy(self.init_capital)
        self._id = -1
        self.__generate()

    def __generate(self):
        for file in os.listdir(self.data_path):
            if file[-3:] == 'npy':
                temp = np.load(os.path.join(self.data_path, file), allow_pickle=True)
                code = file[:6]
                close = temp[4, :]  # 默认featuresmap的第四行为
                date = temp[0]
                data = temp[1:, :]
                sf = self.game(code=code, data=data, date=date, Close=close, windows=self.windows, seed=self._seed,
                               show_statistics=self.show_statistics, init_capital=self.init_capital,**self.kwargs)
                self.envs.append(sf)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def _info(self):
        return self.current_env._info()

    def reset(self, isRandomStart=False, total=None):
        if isRandomStart:
            self._id = self.np_random.randint(len(self.envs))
        else:
            self._id = self._id + 1 if self._id < len(self.envs) - 1 else 0
        if self.show_statistics:
            if total:  # 继续上一次资金,重置附上次统计信息
                self.statistics.append(self.current_env.statistics)
                self.max = self.current_env.max
            else:
                self.statistics = []  # 每次彻底重置都清空统计信息
                self.max = self.init_capital
        self.current_env = self.envs[self._id]
        o = self.current_env.reset(isRandomStart, total, self.max)
        return o

    def step(self, a):
        observations, reward, done, info = self.current_env.step(a)
        return observations, reward, done, info

    def render(self, mode='human'):
        self.current_env.render()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(shape=(shp[:-1] + (shp[-1] * k,)),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        o = np.array(self._get_ob())
        if o.shape[-1] == 1:
            return o.squeeze()
        else:
            return o

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        o = np.array(self._get_ob())
        if o.shape[-1] == 1:
            return o.squeeze(), reward, done, info
        else:
            return o, reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        # return LazyFrames(list(self.frames))
        return LazyFrames(list(self.frames))._frames


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


if __name__ == '__main__':
    import os
    import pandas as pd

    data_path = '/home/zzw/py_work2019/RL/firedup-master/src/Financial_gym/data'
    # game = 'Futures'
    # seed = 123
    # windows = 60
    # show_withdraw = False
    # show_statistics = False
    # cost = 0
    # signal_index = -1
    # # envs_ = assembly_fin(data_path=data_path,
    # #                      game=game,
    # #                      mode=mode,
    # #                      seed=seed,
    # #                      windows=windows,
    # #                      cost=cost,
    # #                      scale=True,
    # #                      signal_index=signal_index)

    # envs_ = Assembly_Fin(data_path='../data/pic/',  # 文件存放路径
    #                      game=continus_Fin_Futures_holding_reward,
    #                      seed=None,
    #                      windows=30,
    #                      init_capital=10000,
    #                      show_statistics=True)

    envs_ = Assembly_Fin_for_pic(data_path='../data/pic/',  # 文件存放路径
                         game=continus_Daily_Fin_Futures_holding_reward_pic,
                         seed=123,
                         windows=30,
                         init_capital=1000000,
                         show_statistics=True)

    np.random.seed(123)

    for j in range(12):
        i = 0
        envs_.reset(isRandomStart=False)
        print(envs_.current_env.code)
        iters = 0
        while True:
            i += 1
            a = np.random.randint(2)

            o, r, done, _info = envs_.step(a)


            if done == 1:  # 资金不足
                print(i)
                break
            elif done == 2:  # 达到索引终点
                iters += envs_.current_env.iters
                envs_.reset(isRandomStart=False, total=envs_.current_env.total)
                print(envs_.current_env.code)
                print(iters)
            elif done == 3:  # 达到回撤限制
                done = 0


    # envs_.reset(isRandomStart=False)
    # print(envs_.current_env._info())
    # acts = [0, 0, 1, 1, 0]
    # for i in range(len(acts)):
    #     o, r, done, _info = envs_.step(acts[i])
    #     print(_info)
    # envs_.reset(isRandomStart=False)
    # acts = [1, 0]
    # for i in range(len(acts)):
    #     o, r, done, _info = envs_.step(acts[i])
    #     print(_info)

    print('end')
    # o = sf.reset()
    #
    # for i in range(10000):
    #     if sf.current_T == sf.terminal:
    #         print("terminal")
    #     o, r, done, _info = sf.step(int(0))
    #
    #     if done == 1:
    #         break
    #
    # print()
