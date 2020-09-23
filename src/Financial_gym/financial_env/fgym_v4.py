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
        raise NotImplemented

    def step(self, action):
        raise NotImplemented

    def reset(self):
        raise NotImplemented

    def render(self, mode='human'):
        raise NotImplemented

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class Fgym_daily_v4(fin_env):
    """按日进行的环境"""

    def __init__(self,
                 code=None,
                 Date=None,
                 Data=None,  # 必须是单独一个合约的数据,非连续主力合约数据
                 seed=None,
                 init_capital=10000,
                 show_statistics=False,
                 cost=-3.4,
                 fold=10,  # 杠杆
                 drawdownrate=0.2,
                 action_set=[-1, 1],
                 ):
        """
        code: 所属合约
        Date:所属日期
        Data:当前的数据,尽量用np.ndarray
        seed:设置随机种子
        init_capital:初始化当前的资金,该资金是紧接着上一天的.
        show_statistics:是否记录统计信息(记录了才可以展示)
        cost:每次操作的成本,<0
        fold:杠杆,跳动一单位表示多少钱,如rb就是10
        drawdoenrate:回撤率,目前未使用
        action_set:操作空间
        """
        self.Date = Date
        self.Close = Data['Close']
        self.Data = Data['data']
        self.Datetime = Data['date']
        self.code = code
        self.len_T = len(self.Datetime)
        self.seed(seed)
        self.action_space = self._action_space(action_set)
        self.observation_space = self._observation_space()

        self.drawdownrate = drawdownrate  # 回撤
        self.fold = fold
        # self.show_withdraw = show_withdraw  # 是否展示回撤，尚未使用
        self.show_statistics = show_statistics  # 是否记录交易信息
        self.act_cost = cost  # 每次操作的成本(对于股票来说卖出才有手续费)
        assert self.act_cost <= 0
        self.init_capital = init_capital
        self.statistics = []
        self.done = None
        self._initialise_capital()

    def _initialise_capital(self, total=None, Max=None, commission=None):
        """每次选中此环境都在self.reset()中初始化该函数"""
        self.current_T = 30  # 初始的交易时刻为30分钟后
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
        return spaces.Box(low=1024, high=1024, dtype=np.float, shape=(self.Data[:, 0].shape))

    def _slicedata(self, slice):
        # 不同数据集可能不一样, 应该只有两种: np.array 或者 []
        return self.Data[:, slice]

    def _statistics(self):  # 只能通过self.reset() 和self.step()调用
        # self.show_statistics == True才调用
        li = dict(code=self.code,
                  DateTime=self.Datetime[self.current_T],
                  rewards=self.reward,
                  current_T=self.current_T,
                  current_price=self.Close[self.current_T],
                  current_position=self.direction,
                  commission=self.commission,
                  total=self.total)
        self.statistics.append(li)

    def reset(self, isRandomStart=False, total=None, Max=None, commission=None):
        self._initialise_capital(total=total, Max=Max, commission=commission)
        self.current_T = 30  # reset是从当天的第30个分钟开始，故时间索引为0
        self.observations = self._slicedata(self.current_T)

        if self.show_statistics:
            self.statistics = []  # 每次重置都清空统计信息
            self._statistics()  # 记录初始的时间,价格以及总资产
        return self.observations

    def _reward(self, action):
        if self.len_T - 2 == self.current_T:  # 到达当天的收盘时刻，所有头寸清空(# TODO 以后可以开发隔日操作)
            self.reward = self.act_cost + self.direction * (
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

        self.Max = self.total if self.Max < self.total else self.Max
        return self.reward

    def _done(self):
        """
        done_dict{0:无状况; 1:资金不足, 2:到达当天收盘, 3:到达合约数据的最后一天, 4:回撤超过最大回撤}
        """
        if self.total < self.Close[self.current_T] * 10 * 0.08:  # 资金不足
            self.done = 1
        elif self.current_T == self.len_T - 1:  # 达到当天终点
            self.done = 2
        elif self.Max / self.total > 1 + self.drawdownrate:  # 回撤超过回撤率限制
            self.done = 3
        else:
            self.done = 0
        return self.done

    def step(self, action):
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

    def _info(self):
        if self.show_statistics == True:
            return self.statistics[-1]  # 返回最后一个,即当前的统计信息


class Fgym_daily_signal(Fgym_daily_v4):
    def __init__(self,
                 code=None,
                 Date=None,
                 Data=None,  # 必须是单独一个合约的数据,非连续主力合约数据
                 seed=None,
                 signal=None,
                 init_capital=10000,
                 show_statistics=False,
                 cost=-3.4,
                 fold=10,  # 杠杆
                 drawdownrate=0.2,
                 action_set=[-1, 1],
                 ):
        """signal:其顺序和个数对应于当前环境的日期"""
        super().__init__(code, Date, Data, seed, init_capital, show_statistics, cost, fold, drawdownrate, action_set)
        self.signal = signal
        self.non0signalindex = np.where(self.signal != 0)[0]  # 保证一定是非0,且都是索引而不是属性值,非0信号的索引
        self.len_non0signal = len(self.non0signalindex)  # 非0信号的长度
        self._Termination = 5  # 强制终止交易时刻,即倒数_Termination分钟停止交易

    def reset(self, isRandomStart=False, total=None, Max=None, commission=None):
        self._initialise_capital(total=total, Max=Max, commission=commission)
        for i in range(self.len_T):
            temp = self.non0signalindex[i]
            if temp > 30:
                self.current_signal_index = i  # 用来记录当前非0信号所在的索引
                self.current_T = temp  # 当前时间的索引
                break
        self.observations = self._slicedata(self.current_T)

        if self.show_statistics:
            self.statistics = []  # 每次重置都清空统计信息
            self._statistics()  # 记录初始的时间,价格以及总资产
        return self.observations

    def step(self, action):
        action = self._action_set[a]
        reward = self._reward(action)
        if self.show_statistics:
            self._statistics()

        self.current_signal_index += 1  # 信号list的当前索引+1
        if self.current_signal_index == self.len_non0signal or self.len_T - self._Termination <= self.non0signalindex[
            self.current_signal_index]:
            # 当强制终止交易时刻在最后一个信号之后,且已经对最后一个做出决策 or 达到当天终点或者到最后5分钟不操作


            self.current_T = self.len_T - self._Termination
            reward += self._reward(action)
            self._statistics()
        else:
            self.current_T = self.non0signalindex[self.current_signal_index]

        done = self._done()
        info = self._info()
        self.observations = self._slicedata(self.current_T)
        # self.iters += 1
        return self.observations, reward, done, info

    def _done(self):
        """
        done_dict{0:无状况; 1:资金不足, 2:到达当天收盘, 3:到达合约数据的最后一天, 4:回撤超过最大回撤}
        """
        if self.total < self.Close[self.current_T] * 10 * 0.08:  # 资金不足
            self.done = 1
        elif self.current_signal_index == self.len_non0signal or self.len_T - self._Termination <= self.non0signalindex[
            self.current_signal_index]:
            # 当强制终止交易时刻在最后一个信号之后,且已经对最后一个做出决策 or 达到当天终点或者到最后5分钟不操作
            self.done = 2
        elif self.Max / self.total > 1 + self.drawdownrate:  # 回撤超过回撤率限制
            self.done = 3
        else:
            self.done = 0
        return self.done

    def _reward(self, action):
        # 到达当天的收盘时刻的前一时刻，所有头寸清空,下一时刻作为终止observation(# TODO 以后可以开发隔日操作)
        if self.len_non0signal == self.current_signal_index or self.len_T - self._Termination <= self.non0signalindex[
            self.current_signal_index]:
            self.reward = self.act_cost + self.direction * (
                    self.Close[self.current_T] - self.last_act_price) * self.fold  # 平仓

            self.direction = 0
            self.total += self.reward
            self.commission += self.act_cost

        else:
            assert self.len_T - 2 > self.current_T  # 出错则是因为超出当天时间的索引
            if action == -1:  # 动作卖出
                if self.direction == 1:
                    self.reward = 2 * self.act_cost + (
                            self.Close[self.current_T] - self.last_act_price) * self.fold  # 平多开空
                    self.total += self.reward
                    self.direction = -1  # 这次操作价格

                    self.last_act_T = self.current_T
                    self.last_act_price = self.Close[self.last_act_T]
                    self.commission += 2 * self.act_cost

                elif self.direction == 0:
                    self.reward = self.act_cost  # 开空
                    self.total += self.reward  # 当前总资产
                    self.direction = -1

                    self.last_act_T = self.current_T
                    self.last_act_price = self.Close[self.last_act_T]
                    self.commission += self.act_cost

                elif self.direction == -1:
                    self.reward = 0  # 不动
                    self.total += self.reward

            elif action == 1:  # 动作买入
                if self.direction == 1:
                    self.reward = 0  # 不动
                    self.total += self.reward

                elif self.direction == 0:
                    self.reward = self.act_cost  # 开多
                    self.total += self.reward
                    self.direction = 1

                    self.last_act_T = self.current_T
                    self.last_act_price = self.Close[self.current_T]
                    self.commission += self.act_cost

                elif self.direction == -1:
                    self.reward = 2 * self.act_cost + self.direction * (
                            self.Close[self.current_T] - self.last_act_price) * self.fold  # 平空开多
                    self.total += self.reward
                    self.direction = 1

                    self.last_act_T = self.current_T
                    self.last_act_price = self.Close[self.current_T]
                    self.commission += 2 * self.act_cost

        self.Max = self.total if self.Max < self.total else self.Max
        return self.reward


class Assembly_Fgym_v4(fin_env):
    def __init__(self,
                 fname=None,  # 存放数据的文件名
                 data_path=None,  # 文件存放路径
                 game=None,
                 seed=123,
                 show_statistics=True,
                 init_capital=1000000,
                 action_set=[-1, 1],
                 cost=-3.4,
                 warm_up=0.1,
                 fold=10,
                 drawdownrate=0.3, ):
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
        self.fold = fold
        self.dradownrate = drawdownrate

        # 每个合约的起始索引
        assert isinstance(warm_up, float)
        self.warm_up = warm_up

        # self.total_length = 0  # 记录总的数据条目数量

        self.statistics = []  # 记录总的统计信息, 非done==1时充值不清0
        self.total = copy.deepcopy(self.init_capital)
        self.Max = copy.deepcopy(self.init_capital)
        self.commission = None

        self.cost = cost
        self._id = -1  # 记录当前正在的第_id个环境，-1表示从未开始过reset
        self._generate()
        self.len_envs = len(self.envs)
        self.done = None

    def _generate(self):
        for code, code_v in self.data.items():
            warm_up = max(round(len(code_v.keys()) * 0.1) - 1, 5)
            i = 0  # 合约的当前日期索引
            for Date, Date_v in code_v.items():
                if i >= warm_up:  # 每个合约的warm_up天数之后才开始转进envs列表
                    # Date_v.keys()
                    sf = self.game(
                        code=code,
                        Date=Date,
                        Data=Date_v,
                        seed=self._seed,
                        init_capital=self.init_capital,
                        show_statistics=self.show_statistics,
                        cost=self.cost,
                        fold=self.fold,  # 杠杆
                        drawdownrate=self.dradownrate,
                        action_set=self.action_set, )

                    self.envs.append(sf)
                i += 1
            # self.total_length += sf.len_T
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        del self.data

    def reset(self, isRandomStart=False):
        if isRandomStart:
            self._id = self.np_random.randint(self.len_envs)  # self._id控制所有合约
        else:
            self._id = self._id + 1 if self._id < self.len_envs - 1 else 0

        self.current_env = self.envs[self._id]

        if self.done == 1 or self.done == None:  # 资金不足or未开始
            if self.show_statistics:
                self.statistics = []
            o = self.current_env.reset(isRandomStart, self.init_capital, self.init_capital, 0)
        elif self.done == 2:  # 当天结束
            if self.show_statistics:
                self.statistics.append(self.current_env.statistics)
            o = self.current_env.reset(isRandomStart, self.total, self.Max, self.commission)
        elif self.done == 3:  # 超过回撤率
            pass
            # if self.show_statistics:
            #     self.statistics.append(self.current_env.statistics)
            # o = self.current_env.reset(isRandomStart, self.total, self.Max, self.commission)
        return o

    def step(self, action):
        observations, reward, done, info = self.current_env.step(a)
        self.done = done
        self.total = self.current_env.total
        self.commission = self.current_env.commission
        self.Max = self.current_env.Max
        return observations, reward, done, info

    def render(self, mode='human'):
        self.current_env.render()


class Assembly_Fgym_signal(Assembly_Fgym_v4):
    def __init__(self,
                 fname=None,  # 存放数据的文件名
                 data_path=None,  # 文件存放路径
                 game=None,
                 seed=123,
                 show_statistics=True,
                 init_capital=1000000,
                 action_set=[-1, 1],
                 cost=-3.4,
                 warm_up=0.1,
                 fold=10,
                 drawdownrate=0.3,
                 ):
        super().__init__(fname, data_path, game, seed, show_statistics, init_capital,
                         action_set, cost, warm_up, fold, drawdownrate)

    def _generate(self):
        for code, code_v in self.data.items():
            warm_up = max(round(len(code_v.keys()) * 0.1) - 1, 5)
            i = 0  # 合约的当前日期索引
            for Date, Date_v in code_v.items():
                signal = Date_v['kdjlabel']
                if i >= warm_up and len(signal) > 70:  # 某天的数据量超过2小时,防止在起始时刻之间没有信号
                    # Date_v.keys()

                    sf = self.game(
                        code=code,
                        Date=Date,
                        Data=Date_v,
                        seed=self._seed,
                        init_capital=self.init_capital,
                        show_statistics=self.show_statistics,
                        cost=self.cost,
                        fold=self.fold,  # 杠杆
                        drawdownrate=self.dradownrate,
                        action_set=self.action_set,
                        signal=signal)

                    self.envs.append(sf)
                i += 1
            # self.total_length += sf.len_T
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        del self.data

if __name__ == '__main__':
    data_path = '../data/preparation_of_RB/'
    fname = 'RB9999.XSGE.bollohlc_singletimelap.pk'

    # # 连续输入观察
    # env = Assembly_Fgym_v4(
    #     fname=fname,
    #     data_path=data_path,
    #     game=Fgym_daily_v4,
    #     seed=123,
    #     show_statistics=True,
    #     init_capital=2000000,
    #     action_set=[-1, 1],
    #     cost=0,
    #         )
    # 按信号输入观察
    env = Assembly_Fgym_signal(
        fname=fname,
        data_path=data_path,
        game=Fgym_daily_signal,
        seed=123,
        show_statistics=True,
        init_capital=1000000,
        action_set=[-1, 1],
        cost=-3.5,
    )

    o = env.reset(isRandomStart=True)
    print('当前合约:', env.current_env.code)
    i = 0
    while True:
        if env._id == env.len_envs - 1:
            print("走到env的最后一天")
        # print(env._id)
        print(i)
        if i == 65926:
            print('stop and review.')
        if env._id == (env.len_envs - 1):
            print('最后一天')
        a = np.random.randint(0, 2)  # 模型做决策
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
            pass
            # env.reset(isRandomStart=False)
        elif done == 4:
            pass
        i += 1
        # if (i+1) %10000 == 0:
        #     print(i)
        # if (env._id+1) % 100 == 0:
        #     print(env._id)
    f = open('RandomStart.pk', 'wb')
    pickle.dump(env.statistics, f)
    f.close()
    print('end')