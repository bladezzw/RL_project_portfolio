from spinup.utils.run_utils import ExperimentGrid
from spinup import fgym_trunk_sac_discrete_v2
import torch

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=2)
    args = parser.parse_args()

    eg = ExperimentGrid(name='sac-pyt')
    from src.Financial_gym.financial_env.fgym import *

    env = lambda: Assembly_Fin_for_pic(data_path='src/Financial_gym/data/pic/',  # 文件存放路径
                                       game=continus_Daily_Fin_Futures_holding_reward_pic,
                                       seed=123,
                                       windows=50,
                                       init_capital=1000000,
                                       show_statistics=True,
                                       drawdown=0.1)

    eg.add('environment', env)
    eg.add('show_kwargs_json', True)
    eg.add('env_name', 'Financial_gym_pic_daily', '', True)

    eg.add('seed', [10 * i for i in range(args.num_runs)])
    eg.add('epochs', 400)
    eg.add('save_freq', 5)  # epoch save frequeece
    eg.add('steps_per_epoch', 400)  # default 4000
    eg.add('start_steps', 500)  # default 10000, start store a=pi(obs)
    eg.add('update_after', 500)  # default 1000, update
    eg.add('use_gpu', True)  # default
    eg.add('gpu_parallel', True)  # default

    eg.add('update_times_every_step', 50)  # default 50
    eg.add('automatic_entropy_tuning', True)  # default
    eg.add('batch_size', 48)  # default
    eg.add('num_test_episodes', 2)  # default


    from spinup.algos.pytorch.sac.core import Discrete_Actor, Discrete_Critic, Xception_1
    eg.add('state_of_art_model', True)
    eg.add('Actor', [Discrete_Actor])
    eg.add('Critic', [Discrete_Critic])
    eg.add('ac_kwargs:model', [Xception_1])
    eg.add('ac_kwargs:num_classes', [2])
    # eg.add('ac_kwargs:hidden_sizes', [(2048, 1024, 512, 256)], 'hid')
    # eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')
    eg.add("last_save_path", "/media/zzw/Magic/py_work2019/RL/spinningup-master/data/sac-pyt_financial_gym_pic_daily/sac-pyt_financial_gym_pic_daily_s0/pyt_save/model.pt")
    eg.run(fgym_trunk_sac_discrete_v2, num_cpu=args.cpu)



