import sys
from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger
from torch.distributions import Categorical
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, device='cpu'):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device
        self._full = False

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        if (self.ptr + 1) % self.max_size == 0:
            self.ptr = 0
            self._full = True
        else:
            self.ptr = (self.ptr + 1) % self.max_size

        self.size = self.max_size if self._full else min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in batch.items()}


def sac_discrete(env_fn, Actor=core.DiscreteMLPActor, Critic=core.DiscreteMLPQFunction, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_times_every_step=50, num_test_episodes=10, max_ep_len=10000,
        logger_kwargs=dict(), save_freq=1, automatic_entropy_tuning=True, use_gpu=False,
        gpu_parallel=False, show_test_render=True, last_save_path=None, **kwargs):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_times_every_step (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        env, test_env = env_fn(), env_fn()
    except:
        env = env_fn
        test_env = deepcopy(env)

    env.seed(seed)
    # env.seed(seed)
    # test_env.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    # Create actor-critic module and target networks
    actor = Actor(obs_dim[0], act_dim, **ac_kwargs)
    critic1 = Critic(obs_dim[0], act_dim, **ac_kwargs)
    critic2 = Critic(obs_dim[0], act_dim, **ac_kwargs)

    critic1_targ = deepcopy(critic1)
    critic2_targ = deepcopy(critic2)
    # gpu是否使用
    if torch.cuda.is_available():
        device = torch.device("cuda" if use_gpu else "cpu")
        if gpu_parallel:
            actor = torch.nn.DataParallel(actor)
            critic1 = torch.nn.DataParallel(critic1)
            critic2 = torch.nn.DataParallel(critic2)
            critic1_targ = torch.nn.DataParallel(critic1_targ)
            critic2_targ = torch.nn.DataParallel(critic2_targ)
    else:
        use_gpu = False
        gpu_parallel = False
        device = torch.device("cpu")
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in critic1_targ.parameters():
        p.requires_grad = False
    for p in critic2_targ.parameters():
        p.requires_grad = False
    actor.to(device)
    critic1.to(device)
    critic2.to(device)
    critic1_targ.to(device)
    critic2_targ.to(device)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=1, size=replay_size, device=device)

    # # List of parameters for both Q-networks (save this for convenience)
    # q_params = itertools.chain(critic1.parameters(), critic2.parameters())

    if automatic_entropy_tuning:
        # we set the max possible entropy as the target entropy
        target_entropy = -np.log((1.0 / act_dim)) * 0.98
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp()
        alpha_optim = Adam([log_alpha], lr=lr, eps=1e-4)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [actor, critic1, critic2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(actor.parameters(), lr=lr)
    q1_optimizer = Adam(critic1.parameters(), lr=lr)
    q2_optimizer = Adam(critic2.parameters(), lr=lr)


    # Set up model saving
    #
    # save_dict = {'epoch': epoch,
    #              'actor': actor.state_dict(),
    #              'critic1': critic1.state_dict(),
    #              'critic2': critic2.state_dict(),
    #              'pi_optimizer': pi_optimizer.state_dict(),
    #              'q1_optimizer': q1_optimizer.state_dict(),
    #              'q2_optimizer': q2_optimizer.state_dict(),
    #              'critic1_targ': critic1_targ.state_dict(),
    #              'critic2_targ': critic2_targ.state_dict(),
    #              }
    # logger.setup_pytorch_saver(save_dict)

    # load_model:
    # load_model:
    if last_save_path is not None:
        checkpoints = torch.load(last_save_path)
        epoch = checkpoints['epoch']
        actor.load_state_dict(checkpoints['actor'])
        critic1.load_state_dict(checkpoints['critic1'])
        critic2.load_state_dict(checkpoints['critic2'])
        pi_optimizer.load_state_dict(checkpoints['pi_optimizer'])
        q1_optimizer.load_state_dict(checkpoints['q1_optimizer'])
        q2_optimizer.load_state_dict(checkpoints['q2_optimizer'])
        critic1_targ.load_state_dict(checkpoints['critic1_targ'])
        critic2_targ.load_state_dict(checkpoints['critic2_targ'])

        # last_best_Return_per_local = checkpoints['last_best_Return_per_local']
        print("succesfully load last prameters")
    else:
        epoch = 0

        print("Dont load last prameters.")

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):

        # Bellman backup for Q functions
        with torch.no_grad():
            o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
            if r.ndim == 1:
                r = r.unsqueeze(-1)
            if d.ndim == 1:
                d = d.unsqueeze(-1)
            # Target actions come from *current* policy
            a2, (a2_p, logp_a2), _ = get_action(o2)

            # Target Q-values
            q1_pi_targ = critic1_targ(o2)
            q2_pi_targ = critic2_targ(o2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            min_qf_next_target = a2_p * (q_pi_targ - alpha * logp_a2)
            min_qf_next_target = min_qf_next_target.mean(dim=1).unsqueeze(-1)
            backup = r + gamma * (1 - d) * min_qf_next_target

        q1 = critic1(o).gather(1, a.long())
        q2 = critic2(o).gather(1, a.long())
        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q1, loss_q2, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        state_batch = data['obs']
        action, (action_probabilities, log_action_probabilities), _ = get_action(state_batch)
        qf1_pi = critic1(state_batch)
        qf2_pi = critic2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = alpha * log_action_probabilities - min_qf_pi
        policy_loss = action_probabilities * inside_term
        policy_loss = policy_loss.mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        # Useful info for logging
        pi_info = dict(LogPi=log_action_probabilities.detach().cpu().numpy())

        return policy_loss, log_action_probabilities, pi_info

    def take_optimisation_step(optimizer, network, loss, clipping_norm=None, retain_graph=False):
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad()  # reset gradients to 0
        loss.backward(retain_graph=retain_graph)  # this calculates the gradients
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               clipping_norm)  # clip gradients to help stabilise training
        optimizer.step()  # this applies the gradients

    def soft_update_of_target_network(local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update(data):
        # First run one gradient descent step for Q1 and Q2

        loss_q1, loss_q2, q_info = compute_loss_q(data)
        take_optimisation_step(q1_optimizer, critic1, loss_q1, 5, )
        take_optimisation_step(q2_optimizer, critic2, loss_q2, 5, )

        # Record things
        logger.store(LossQ=(loss_q1.item()+loss_q2.item())/2., **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # # computing gradients for them during the policy learning step.
        # for p in q_params:
        #     p.requires_grad = False

        # Next run one gradient descent step for pi.

        loss_pi, log_pi, pi_info = compute_loss_pi(data)
        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # # Unfreeze Q-networks so you can optimize it at next DDPG step.
        # for p in q_params:
        #     p.requires_grad = True

        if automatic_entropy_tuning:
            alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
            # logger.store(alpha_loss=alpha_loss.item())

        take_optimisation_step(pi_optimizer, actor, loss_pi, 5, )

        with torch.no_grad():
            for p, p_targ in zip(critic1.parameters(), critic1_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(critic2.parameters(), critic2_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        if automatic_entropy_tuning:
            take_optimisation_step(alpha_optim, None, alpha_loss, None)
            alpha = log_alpha.exp()

    def get_action(state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities = actor(state)
        max_probability_action = torch.argmax(action_probabilities).unsqueeze(0)
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action


    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                if show_test_render: test_env.render()
                # Take deterministic actions at test time
                with torch.no_grad():
                    _, (_, _), a = get_action(torch.FloatTensor([o]).to(device))
                o, r, d, _ = test_env.step(a.cpu().item())
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    eps = 1

    t = epoch*steps_per_epoch if last_save_path is not None else 0

    # Main loop: collect experience in env and update/log each epoch
    actor.eval()
    while t < total_steps:

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t >= start_steps:
            with torch.no_grad():
                a, _, _ = get_action(torch.FloatTensor([o]).to(device)) if o.shape == obs_dim else get_action(torch.FloatTensor(o).to(device))
                a = a.cpu().item()
        else:
            a = np.random.randint(0, act_dim)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2


        # End of trajectory handling
        if d or (ep_len == max_ep_len):  # ep_len == max_ep_len是游戏成功时最少ep长度
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            text = "\r\x1b[32mEpoch: %s,  Episode: %s,  Ep_ret: %s,  ep_len: %s. [%s/%s] \x1b[0m" % \
                   (epoch, eps, ep_ret, ep_len, t+1,  total_steps)
            sys.stdout.write(text)
            sys.stdout.flush()
            o, ep_ret, ep_len = env.reset(), 0, 0
            # if eps % 30 == 0:
            #     logger.log('\nEpisode: %s\n,\tEp_ret: %s,\tep_len: %s' % (eps, ep_ret,ep_len))
            eps += 1

        # Update handling
        if t >= update_after and t % update_times_every_step == 0:
            actor.train()
            for j in range(update_times_every_step):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)
                # Save model
            actor.eval()
            # logger.save_epoch_Ret_optimizer_model(save_dict)
            # last_best_Return_per_local = Return_per_local
        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0 and t > update_after:  # steps_perepoch步 and 大于update_after步
            if (t+1) % update_times_every_step == 0:  # 每达到update_times_every_step
                epoch = (t + 1) // steps_per_epoch

                if proc_id() == 0 and (epoch) % save_freq == 0:
                    save_dict = {'epoch': epoch,
                                 'actor': actor.state_dict(),
                                 'critic1': critic1.state_dict(),
                                 'critic2': critic2.state_dict(),
                                 'pi_optimizer': pi_optimizer.state_dict(),
                                 'q1_optimizer': q1_optimizer.state_dict(),
                                 'q2_optimizer': q2_optimizer.state_dict(),
                                 'critic1_targ': critic1_targ.state_dict(),
                                 'critic2_targ': critic2_targ.state_dict(),
                                 }
                    logger.save_epoch_Ret_optimizer_model(save_dict, epoch)

                actor.eval()
                # Test the performance of the deterministic version of the agent.
                test_agent()

                # Log info about epoch
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('TestEpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('TestEpLen', average_only=True)
                logger.log_tabular('TotalEnvInteracts', t)
                logger.log_tabular('Q1Vals', with_min_and_max=True)
                logger.log_tabular('Q2Vals', with_min_and_max=True)
                logger.log_tabular('LogPi', with_min_and_max=True)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('LossQ', average_only=True)
                logger.log_tabular('Time', time.time() - start_time)
                # if epoch > 1:
                #     (time.time() - start_time)/epo
                logger.dump_tabular()

        t += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac_discrete(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
