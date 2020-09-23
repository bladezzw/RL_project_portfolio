# from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_portfolio
import torch

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    from environments.rlstock.rlstock_env_window_features import StockEnv
    # from environments.rlstock.rlstock_testenv_window_features import StockTestEnv
    import spinup.algos.pytorch.ppo.core_portfolio as core
    ppo_portfolio(env_fn=lambda: StockEnv(),
                actor_critic=core.MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128, 256, 64], activation=torch.nn.Tanh),
                seed=0, steps_per_epoch=5000, epochs=200, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
                vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=5000,
                target_kl=0.01,
                logger_kwargs=dict(output_dir='data/ppo_rlstock/', output_fname='progress.txt', exp_name='tanh'),
                save_freq=10)

    # ddpg_pytorch(env_fn=lambda: StockEnv(),
    #              test_env=lambda: StockTestEnv(),
    #              show_kwargs_json=True,
    #              logger_kwargs=dict(output_dir='data/ddpg_rlstock/', output_fname='progress.txt', exp_name='tanh'),
    #              num_test_episodes=1,
    #              epochs=400,
    #              steps_per_epoch=4000,
    #              ac_kwargs=dict(hidden_sizes=[128, 128], activation=torch.nn.Tanh),
    #              seed=0,
    #              replay_size=int(1e6),
    #              gamma=0.99,
    #              polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
    #              update_after=1000, update_every=50, act_noise=0.1,
    #              max_ep_len=1000, save_freq=1
    #              )
