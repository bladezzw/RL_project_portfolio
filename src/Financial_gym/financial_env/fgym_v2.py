import os
import numpy as np
import copy
import pickle
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


class Futures_holding_reward_pic(Env):
    # Depricated
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
        self.date = date  # 日期和小时分钟
        self.Date = [d[:10] for d in self.date]  # 日期
        self.Time = [d[11:] for d in self.date]  # 小时分钟
        if Close is None:
            self.Close = self.data[3, :]
        else:
            self.Close = Close
        self.len_T = len(self.Time)
        self.warm_up = warm_up if warm_up else round(self.len_T * 0.1)  # 预热

        self.seed(seed)
        # w, h, =   # width height channel
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(self.data[0].shape))
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

    def _initialise_capital(self, total=None, max=None, commission=None):
        self.iters = 0
        self.current_act_T = None  # 这次操作的时间索引
        self.current_act_price = None  # 这次操作价格
        self.total = total if total else copy.deepcopy(self.init_capital)  # 如果给定新的总资产, 否则默认总资产
        self.last_act_T = None  # 上次操作的时间索引,(操作有买入/卖出)
        self.last_act_price = None  # 上次操作的价格
        self.commission = commission if commission else 0
        self.num_act = 1  # 每次操作的数量
        self.direction = 0  # 持有的头寸方向

        self.reward = 0
        self.max = max if max else copy.deepcopy(self.init_capital)

    def _reward(self, action):
        if action == -1:  # 动作卖出
            if self.direction == 1:
                self.reward = 2 * self.act_cost + (
                        self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                self.total += self.reward
                self.direction = -1  # 这次操作价格
                self.commission += 2 * self.act_cost

            elif self.direction == 0:
                self.reward = self.act_cost  # 收益
                self.total += self.reward  # 当前总资产
                self.direction = -1
                self.commission += self.act_cost

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
                self.commission += self.act_cost

            elif self.direction == -1:
                self.reward = 2 * self.act_cost + self.direction * (
                        self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                self.total += self.reward
                self.direction = 1
                self.commission += 2 * self.act_cost

        self.max = self.total if self.max < self.total else self.max
        return self.reward

    def _done(self):  # TODO:逻辑不是很合理,有待修改
        if self.total < self.Close[self.current_T] * 10 * 0.08:  # 资金不足

            return 1
        elif self.current_T == self.len_T - 1:  # 达到索引终点
            return 2
        elif self.max / self.total > 1 + self.drawdown:  # 回撤超过回撤率限制
            return 3
        else:
            return 0

    def _info(self):
        if self.show_statistics == True:
            return self.statistics[-1]  # 返回最后一个,即当前的统计信息

    def _statistics(self):  # 只能通过self.reset() 和self.step()调用
        # self.show_statistics == True才调用

        li = dict(code=self.code,
                  DateTime=self.date[self.current_T],
                  rewards=self.reward,
                  current_T=self.current_T,
                  current_price=self.Close[self.current_T],
                  current_position=self.direction,
                  commission=self.commission,
                  total=self.total)
        self.statistics.append(li)

    def step(self, a):
        action = self._action_set[a]
        reward = self._reward(action)

        if self.show_statistics:
            self._statistics()
        self.current_T += 1
        done = self._done()
        info = self._info()

        self.observations = self.data[self.current_T]
        self.iters += 1

        return self.observations, reward, done, info

    def reset(self, isRandomStart=False, total=None, max=None, commission=None):
        """
        isRandomStart: 从环境的开始走到结尾/随机开始
        :return: observation (object): agent's observation of the current environment
        """
        self._initialise_capital(total, max, commission)
        self.start_T = self.np_random.randint(self.warm_up, self.len_T) if isRandomStart else self.warm_up
        self.current_T = copy.deepcopy(self.start_T)
        self.observations = self.data[self.start_T]
        if self.show_statistics:
            self.statistics = []  # 每次重置都清空统计信息
            self._statistics()  # 记录初始的时间,价格以及总资产
        return self.observations

    def render(self, mode='human'):
        Image.fromarray(self.observations).show()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class Daily_Futures_holding_reward_pic(Futures_holding_reward_pic):
    # Depricated
    def __init__(self,
                 code=None,
                 data=None,  # 必须是单独一个合约的数据,非连续主力合约数据
                 date=None,
                 Close=None,
                 seed=None,
                 data_shape=[],
                 warm_up=None,
                 init_capital=10000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=-3.4,
                 fold=10,  # 杠杆
                 start_Time='09:32:00',
                 **kwargs,
                 ):
        super(Daily_Futures_holding_reward_pic, self).__init__(
            code=code,
            data=data,  # 必须是单独一个合约的数据,非连续主力合约数据
            date=date,
            Close=Close,
            seed=seed,
            data_shape=data_shape,
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
        self.checked_date = []  # 使用(用来计算过)过的交易日.

    def join_date_time(self, date='', time=''):
        return date + ' ' + time

    def _reward(self, action):

        if self.current_T + 2 == self.len_T:  # 合约结束
            self.reward = self.act_cost + self.direction * (
                    self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
            self.total += self.reward
            self.direction = 0
            self.commission += self.act_cost
        # 当天收盘
        elif (self.Time[self.current_T] < '02:00:00' and self.Time[self.current_T + 1] > '09:00:00'):
            self.reward = self.act_cost + self.direction * (
                    self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
            self.total += self.reward
            self.direction = 0
            self.commission += self.act_cost
        # 当天收盘
        elif((self.Date[self.current_T] < self.Date[self.current_T + 1])and self.Time[self.current_T + 1]>='09:00:00'):
            self.reward = self.act_cost + self.direction * (
                    self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
            self.total += self.reward
            self.direction = 0
            self.commission += self.act_cost
        else:
            if action == -1:  # 动作卖出
                if self.direction == 1:
                    self.reward = 2 * self.act_cost + (
                            self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                    self.total += self.reward
                    self.direction = -1  # 这次操作价格
                    self.commission += 2 * self.act_cost

                elif self.direction == 0:
                    self.reward = self.act_cost  # 收益
                    self.total += self.reward  # 当前总资产
                    self.direction = -1
                    self.commission += self.act_cost

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
                    self.commission += self.act_cost

                elif self.direction == -1:
                    self.reward = 2 * self.act_cost + self.direction * (
                            self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
                    self.total += self.reward
                    self.direction = 1
                    self.commission += 2 * self.act_cost
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
            None if self.next_date in self.checked_date else self.checked_date.append(self.next_date)
        else:
            self.current_T += 1
        info = self._info()
        done = self._done()

        self.iters += 1
        self.observations = self.data[self.current_T]
        return self.observations, reward, done, info

    def reset(self, isRandomStart=False, total=None, max=None, commission=None):
        self._initialise_capital(total, max, commission)

        self.start_date = self.date_sorted[
            self.np_random.randint(self.warm_up, len(self.date_sorted))] if isRandomStart else \
            self.date_sorted[self.warm_up]  # 开始日期,

        self.start_datetime = self.join_date_time(self.start_date, self.start_Time)
        try:
            self.start_T = np.where(self.date == self.start_datetime)[0].item()
        except:
            self.observations = self.reset(isRandomStart, total, max)
            return self.observations
        self.current_T = self.start_T
        self.observations = self.data[self.start_T]
        if self.show_statistics:
            self.statistics = []  # 每次重置都清空统计信息
            self._statistics()  # 记录初始的时间,价格以及总资产
        return self.observations


class Assembly_Fin_for_pic_v2(Env):
    def __init__(self, data_path='../data/',  # 文件存放路径
                 game=None,
                 seed=123,
                 show_statistics=True,
                 init_capital=1000000,
                 **kwargs):
        self._seed = seed
        self.seed(self._seed)  # 随机选择环境的seed和每个环境中的seed不一样
        self.envs = []
        self.data_path = data_path
        self.game = game
        self.show_statistics = show_statistics
        self.init_capital = init_capital
        self.kwargs = kwargs
        self.total_length = 0
        self.statistics = []  # 记录总的统计信息, 非done==1时充值不清0
        self.total = copy.deepcopy(self.init_capital)
        self.max = copy.deepcopy(self.init_capital)
        self._id = -1
        self.__generate()

    def __generate(self):
        for file in os.listdir(self.data_path):
            if file[-2:] == "pk":
                f = open(os.path.join(self.data_path, file), 'rb')
                data = pickle.load(f)
                code = file[:6]
                close = data['Close']  # 默认featuresmap的第四行为
                date = data['Date']
                data0 = data['pic']
                f.close()
                del data
                sf = self.game(code=code, data=data0, date=date, Close=close, seed=self._seed,
                               show_statistics=self.show_statistics, init_capital=self.init_capital, **self.kwargs)
                self.envs.append(sf)
                self.total_length += sf.len_T
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    def _info(self):
        return self.current_env._info()

    def reset(self, isRandomStart=False, total=None, commission=None):
        if isRandomStart:
            self._id = self.np_random.randint(len(self.envs))  # self._id控制所有合约
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
        o = self.current_env.reset(isRandomStart, total, self.max, commission)
        return o

    def step(self, a):
        observations, reward, done, info = self.current_env.step(a)
        return observations, reward, done, info

    def render(self, mode='human'):
        self.current_env.render()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == '__main__':

    # data_path = '/home/zzw/py_work2019/RL/firedup-master/src/Financial_gym/data'

    def trunk(a=1):
        envs_ = Assembly_Fin_for_pic_v2(data_path='../data/pic_data/',  # 文件存放路径
                                        game=Daily_Futures_holding_reward_pic,
                                        seed=123+os.getpid(),
                                        init_capital=10000,
                                        show_statistics=True)

        np.random.seed(123+os.getpid())
        Duration = []
        # if 'Duration.pk' in os.listdir():
        #     f = open('Duration.pk', 'rb')
        #     Duration = pickle.load(f)
        #     f.close()
        # Image.fromarray(o[:, :, :]).show()
        for j in range(100000):
            i = 0
            o = envs_.reset(isRandomStart=True)
            print('当前合约代码:', envs_.current_env.code)
            iters = 0

            print('第', j,'次')
            # if j == 21:
            #     print(j)

            while True:
                i += 1
                a = np.random.randint(2)

                o, r, done, _info = envs_.step(a)

                if done == 1:  # 资金不足
                    print(i)
                    Duration.append(i)
                    break
                elif done == 2:  # 达到索引终点
                    iters += envs_.current_env.iters
                    envs_.reset(isRandomStart=True, total=envs_.current_env.total)
                    # print(envs_.current_env.code)
                    # print(iters)
                elif done == 3:  # 达到回撤限制
                    done = 0
        print(len(Duration))
        f = open('Duration.pk'+str(os.getpid()), 'wb')
        pickle.dump(Duration, f)
        f.close()
        print('end')

    # 多线程生成累计长度
    from multiprocessing import Pool
    p = Pool(8)
    for i in range(8):

        p.apply_async(trunk, args=(1,))

    p.close()  # 关闭进程池，用户不能再向这个池中提交任务了
    p.join()