from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    # define environment
    eg = ExperimentGrid(name='ppo-pyt-bench')
    from src.Financial_gym.financial_env.fgym import Assembly_Fin, continus_Fin_Futures_holding_reward
    env_id = 'holding_reward'
    env = Assembly_Fin(data_path='/home/zzw/py_work2019/RL/spinningup-master/src/Financial_gym/data/Clipped',  # 文件存放路径
                         game=continus_Fin_Futures_holding_reward,
                         seed=None,
                         windows=30,
                         init_capital=10000,
                         show_statistics=True)

    eg.add('environment', env)
    eg.add('show_kwargs_json', False)
    eg.add('env_name', env_id, '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 400)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(64, 64)], 'hid')
    # eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')
    eg.run(ppo_pytorch, num_cpu=args.cpu)