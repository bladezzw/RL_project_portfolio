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
        self.observation_space = spaces.Box(low=-1024, high=1024, dtype=np.float, shape=(data_shape[0], windows))

        pass

    def step(self, action):
        raise NotImplemented

    def reset(self):
        raise NotImplemented

    def render(self, mode='human'):
        raise NotImplemented

    def seed(self, seed=None):
        super(fin_env, self).seed(seed)

class Fgym_daily_v4(Env):
    def __init__(self,
                 code=None,
                 Date=None,
                 Contract=None,
                 data=None,  # 必须是单独一个合约的数据,非连续主力合约数据
                 seed=None,
                 init_capital=10000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=-3.4,
                 fold=10,  # 杠杆
                 drawdownrate=0.2,
                 action_set=[-1, 1],
                 ):
        self.data = data
        self.DateTime = Date
        self.code = code
        self.Contract = Contract
        self.seed(seed)
        self.action_space = self._action_space(action_set)
        self.observation_space = self._observation_space()

        self.drawdownrate = drawdownrate  # 回撤
        self.fold = fold
        self.show_withdraw = show_withdraw  # 是否展示回撤，尚未使用
        self.show_statistics = show_statistics  # 是否记录交易信息
        self.act_cost = cost  # 每次操作的成本(对于股票来说卖出才有手续费)
        assert self.act_cost < 0
        self.init_capital = init_capital
        self.statistics = []
        self.done = None
        self._initialise_capital()

    def _initialise_capital(self, total=None, Max=None, commission=None):
        """每次选中此环境都在self.reset()中初始化该函数"""
        self.current_T = 30  # 初始的交易时刻
        self.current_act_T = None  # 这次操作的时间索引
        self.current_act_price = None  # 这次操作价格
        self.total = total if total else copy.deepcopy(self.init_capital)  # 如果给定新的总资产, 否则默认总资产
        self.last_act_T = None  # 上次操作的时间索引,(操作有买入/卖出)
        self.last_act_price = None  # 上次操作的价格
        self.commission = commission if commission else 0
        self.num_act = 1  # 每次操作的数量
        self.direction = 0  # 持有的头寸方向
        self.reward = 0
        self.Max = Max if Max else copy.deepcopy(self.init_capital)

    def _action_space(self, action_set=[-1, 1]):
        self._action_set = action_set  #
        return spaces.Discrete(len(self._action_set))

    def _observation_space(self):
        """不同的数据集要重写"""
        return spaces.Box(low=-1024, high=1024, dtype=np.float, shape=(self.data[:,0].shape))

class Assembly_Fgym_v4(Env):
    def __init__(self,
                 fname=None,  # 存放数据的文件名
                 data_path=None,  # 文件存放路径
                 game=None,
                 seed=123,
                 show_statistics=True,
                 init_capital=1000000,
                 action_set=[-1, 1],
                 **kwargs):
        dirfname = os.path.join(data_path, fname)
        f = open(dirfname, 'rb')
        self.data = pickle.load(f)
        f.close()

        self._seed = seed
        self.seed(self._seed)  # 随机选择环境的seed和每个环境中的seed不一样
        self.envs = []

        self.game = game
        self.show_statistics = show_statistics
        self.init_capital = init_capital
        self.action_set = action_set
        self.kwargs = kwargs
        # self.total_length = 0  # 记录总的数据条目数量
        self.statistics = []  # 记录总的统计信息, 非done==1时充值不清0
        self.total = copy.deepcopy(self.init_capital)
        self.Max = copy.deepcopy(self.init_capital)
        self._id = -1  # 记录当前正在的第_id个环境，-1表示从未开始过reset
        self.__generate()
        self.done = None
        self.commission = None

    def __generate(self):
        for code, code_v in self.data.items():

            sf = self.game(code=code, data=code_v, seed=self._seed, action_set=self.action_set,
                           show_statistics=self.show_statistics, init_capital=self.init_capital, **self.kwargs)

            self.envs.append(sf)
            # self.total_length += sf.len_T
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def _info(self):
        return self.current_env._info()

    def reset(self, isRandomStart=False):
        done = self.done
        if done == 1 or done is None:  # 没钱了
            if isRandomStart:
                self._id = self.np_random.randint(len(self.envs))  # self._id控制所有合约
            else:
                self._id = self._id + 1 if self._id < len(self.envs) - 1 else 0
            if self.show_statistics:
                    self.statistics = []  # 每次彻底重置都清空统计信息
            self.current_env = self.envs[self._id]
            o = self.current_env.reset(isRandomStart)
        elif done == 2:  # 运行完一个交易日的数据
            o = self.current_env.reset(isRandomStart=isRandomStart, total=self.total, Max=self.Max, commission=self.commission)
        else:  # done = 3,4
            if isRandomStart:
                self._id = self.np_random.randint(len(self.envs))  # self._id控制所有合约
            else:
                self._id = self._id + 1 if self._id < len(self.envs) - 1 else 0
            if self.show_statistics:
                self.statistics.append(self.current_env.statistics)
            self.current_env = self.envs[self._id]
            o = self.current_env.reset(isRandomStart, self.total, self.Max, self.commission)
        return o

    def step(self, a):
        observations, reward, done, info = self.current_env.step(a)
        self.done = done
        self.total = self.current_env.total
        self.commission = self.current_env.commission
        self.Max = self.current_env.Max
        return observations, reward, done, info

    def render(self, mode='human'):
        self.current_env.render()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class Fgym_daily(Env):
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
                 seed=None,
                 warm_up=None,
                 init_capital=10000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=-3.4,
                 fold=10,  # 杠杆
                 drawdownrate=0.2,
                 action_set=[-1, 1],
                 ):
        """期货交易,双向盈利,持续交易,持有时有反馈"""


        if data is None:
            raise ValueError
        self.code = code
        self.data = data
        self.Date = list(self.data.keys())  # 日期
        # self.Time = [d[11:] for d in self.date]  # 小时分钟

        self.len_D = len(self.Date)
        # 预热从多个日期(self.Date)中选择起始的日期,索引要-1
        self.warm_up = warm_up if warm_up else max(3, round(self.len_D * 0.1)-1)
        self.seed(seed)
        # w, h, =   # width height channel
        self.action_space = self._action_space(action_set)
        self.observation_space = self._observation_space()
        self.drawdownrate = drawdownrate  # 回撤
        self.fold = fold
        self.show_withdraw = show_withdraw  # 是否展示回撤，尚未使用
        self.show_statistics = show_statistics  # 是否记录交易信息
        self.init_capital = init_capital  # 初始资金
        self.act_cost = cost  # 每次操作的成本(对于股票来说卖出才有手续费)
        assert self.act_cost < 0

        self.statistics = []
        # financial affairs
        self.start_D = None  # reset()之后的开始日期索引
        self.current_D = None  # 当前日期的索引
        self.current_T = -1  # 当前的时间索引,-1表示从未开始过reset
        self.done = None
        self.Date_data = None  # 当前的日期的数据
        self._initialise_capital()
    def _action_space(self, action_set=[-1, 1]):
        self._action_set = action_set  #
        return spaces.Discrete(len(self._action_set))

    def _observation_space(self):
        """不同的数据集要重写"""
        return spaces.Box(low=-1024, high=1024, dtype=np.float, shape=(self.data[self.Date[0]]['data'][:,0].shape))

    def _initialise_capital(self, total=None, Max=None, commission=None):
        # self.iters = 0  # 记录当前环境已step过的数据条目数量
        self.current_act_T = None  # 这次操作的时间索引
        self.current_act_price = None  # 这次操作价格
        self.total = total if total else copy.deepcopy(self.init_capital)  # 如果给定新的总资产, 否则默认总资产
        self.last_act_T = None  # 上次操作的时间索引,(操作有买入/卖出)
        self.last_act_price = None  # 上次操作的价格
        self.commission = commission if commission else 0
        self.num_act = 1  # 每次操作的数量
        self.direction = 0  # 持有的头寸方向

        self.reward = 0
        self.Max = Max if Max else copy.deepcopy(self.init_capital)

    def _reward(self, action):
        if self.len_T-2 == self.current_T:  # 到达当天的收盘时刻，所有头寸清空(# TODO 以后可以开发隔日操作)
            self.reward = self.act_cost + self.direction*(
                    self.Close[self.current_T] - self.Close[self.current_T - 1]) * self.fold
            self.total += self.reward
            self.direction = 0
            self.commission += self.act_cost

        else:
            assert self.len_T - 2 > self.current_T  # 出错则是因为超出当天时间的索引
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

        self.Max = self.total if self.Max < self.total else self.Max
        return self.reward

    def _done(self):
        """
        done_dict{0:无状况; 1:资金不足, 2:到达当天收盘, 3:到达合约数据的最后一天, 4:回撤超过最大回撤}
        """
        if self.total < self.Close[self.current_T] * 10 * 0.08:  # 资金不足
            self.done = 1
            self.current_D = None  # 当前日期的索引

        elif self.current_T == self.len_T - 1:  # 达到当天终点
            if self.current_D == self.len_D - 1:  # 到达合约终点
                self.done = 3
                self.current_D = None  # 当前日期的索引
                return self.done
            self.done = 2
        elif self.Max / self.total > 1 + self.drawdownrate:  # 回撤超过回撤率限制
            self.done = 4
        else:
            self.done = 0
        return self.done

    def _info(self):
        if self.show_statistics == True:
            return self.statistics[-1]  # 返回最后一个,即当前的统计信息

    def _statistics(self):  # 只能通过self.reset() 和self.step()调用
        # self.show_statistics == True才调用
        li = dict(code=self.code,
                  DateTime=self.Date_data['date'][self.current_T],
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

        self.observations = self._slicedata(self.current_T)
        # self.iters += 1

        return self.observations, reward, done, info

    def _slicedata(self, slice):
        # 不同数据集可能不一样, 应该只有两种: np.array 或者 []
        # slice = self.Date_data['date'][slice] if isinstance(self.Date_data['data'], dict) else slice
        return self.Date_data['data'][:, slice]

    def reset(self, isRandomStart=False, total=None, Max=None, commission=None):
        """
        isRandomStart: 从环境的开始走到结尾/随机开始
        :return: observation (object): agent's observation of the current environment
        """
        self._initialise_capital(total, Max, commission)
        self.start_D = self.np_random.randint(self.warm_up, self.len_D) if isRandomStart else (self.warm_up)
        self.start_Date = self.Date[self.start_D]


        self.current_T = 30  # reset是从当天的第30个时刻开始，故时间索引为0
        # 如果不是第一次则日期+1, done==3不跳出就重新从self.warm_up开始
        self.current_D = self.start_D if not self.current_D else self.current_D + 1

        self.Date_data = self.data[self.Date[self.current_D]]
        self.Close = self.Date_data['Close']
        self.len_T = len(self.Date_data['date'])
        self.observations = self._slicedata(self.current_T)
        if self.show_statistics and self.done == None:
            self.statistics = []  # 每次重置都清空统计信息
            self._statistics()  # 记录初始的时间,价格以及总资产
        else:
            self._statistics()  # 记录初始的时间,价格以及总资产
        return self.observations

    def render(self, mode='human'):
        Image.fromarray(self.observations).show()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class Fgym_daily_signal(Fgym_daily):
    def __init__(self,
                 code=None,
                 data=None,  # 必须是单独一个合约的数据,非连续主力合约数据
                 seed=None,
                 warm_up=None,
                 init_capital=10000,
                 show_withdraw=False,
                 show_statistics=False,
                 cost=-3.4,
                 fold=10,  # 杠杆
                 drawdownrate=0.2,
                 action_set=[-1, 1],
                 ):
        super().__init__(
                 code=code,
                 data=data,  # 必须是单独一个合约的数据,非连续主力合约数据
                 seed=seed,
                 warm_up=warm_up,
                 init_capital=init_capital,
                 show_withdraw=show_withdraw,
                 show_statistics=show_statistics,
                 cost=cost,
                 fold=fold,  # 杠杆
                 drawdownrate=drawdownrate,
                 action_set=action_set,
                 )



class Assembly_Fgym_v3(Env):
    def __init__(self,
                 fname=None,  # 存放数据的文件名
                 data_path=None,  # 文件存放路径
                 game=None,
                 seed=123,
                 show_statistics=True,
                 init_capital=1000000,
                 action_set=[-1, 1],
                 **kwargs):
        dirfname = os.path.join(data_path, fname)
        f = open(dirfname, 'rb')
        self.data = pickle.load(f)
        f.close()

        self._seed = seed
        self.seed(self._seed)  # 随机选择环境的seed和每个环境中的seed不一样
        self.envs = []

        self.game = game
        self.show_statistics = show_statistics
        self.init_capital = init_capital
        self.action_set = action_set
        self.kwargs = kwargs
        # self.total_length = 0  # 记录总的数据条目数量
        self.statistics = []  # 记录总的统计信息, 非done==1时充值不清0
        self.total = copy.deepcopy(self.init_capital)
        self.Max = copy.deepcopy(self.init_capital)
        self._id = -1  # 记录当前正在的第_id个环境，-1表示从未开始过reset
        self.__generate()
        self.done = None
        self.commission = None

    def __generate(self):
        for code, code_v in self.data.items():

            sf = self.game(code=code, data=code_v, seed=self._seed, action_set=self.action_set,
                           show_statistics=self.show_statistics, init_capital=self.init_capital, **self.kwargs)

            self.envs.append(sf)
            # self.total_length += sf.len_T
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def _info(self):
        return self.current_env._info()

    def reset(self, isRandomStart=False):
        done = self.done
        if done == 1 or done is None:  # 没钱了
            if isRandomStart:
                self._id = self.np_random.randint(len(self.envs))  # self._id控制所有合约
            else:
                self._id = self._id + 1 if self._id < len(self.envs) - 1 else 0
            if self.show_statistics:
                    self.statistics = []  # 每次彻底重置都清空统计信息
            self.current_env = self.envs[self._id]
            o = self.current_env.reset(isRandomStart)
        elif done == 2:  # 运行完一个交易日的数据
            o = self.current_env.reset(isRandomStart=isRandomStart, total=self.total, Max=self.Max, commission=self.commission)
        else:  # done = 3,4
            if isRandomStart:
                self._id = self.np_random.randint(len(self.envs))  # self._id控制所有合约
            else:
                self._id = self._id + 1 if self._id < len(self.envs) - 1 else 0
            if self.show_statistics:
                self.statistics.append(self.current_env.statistics)
            self.current_env = self.envs[self._id]
            o = self.current_env.reset(isRandomStart, self.total, self.Max, self.commission)
        return o

    def step(self, a):
        observations, reward, done, info = self.current_env.step(a)
        self.done = done
        self.total = self.current_env.total
        self.commission = self.current_env.commission
        self.Max = self.current_env.Max
        return observations, reward, done, info

    def render(self, mode='human'):
        self.current_env.render()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

if __name__ == '__main__':
    data_path = '../data/preparation_of_RB/'
    fname = 'RB9999.XSGE.bollohlc_singletimelap.pk'

    env = Assembly_Fgym_v3(
            fname=fname,
            data_path=data_path,
            game=Fgym_daily,
            seed=123,
            show_statistics=True,
            init_capital=1000000,
            action_set=[-1, 1])

    o = env.reset(isRandomStart=True)
    print('当前合约:', env.current_env.code)
    i = 0
    while True:
        a = np.random.randint(0, 2)
        observations, reward, done, info = env.step(a)
        if done == 0:
            pass
        elif done == 1:
            break
            # env.reset()
        elif done == 2:
            env.reset(isRandomStart=True)
            # print('当前合约:', env.current_env.code)
        elif done == 3:
            env.reset(isRandomStart=True)
        elif done == 4:
            pass
        i += 1
        if (i+1) %10000 == 0:
            print(i)
    f = open('RandomStart.pk','wb')
    pickle.dump(env.statistics,f)
    f.close()
    print('end')