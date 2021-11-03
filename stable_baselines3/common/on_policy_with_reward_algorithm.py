import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer, EntReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv
from collections import deque
from logger import Logger

import utils

class OnPolicyRewardAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        reward_model,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        num_interaction: int = 5000, 
        num_feed: int = 1,
        feed_type: int = 0,
        re_update: int = 100,
        re_large_batch: int = 2,
        metaworld_flag: bool = False,
        max_feed: int = 1400,
        size_segment: int = 25,
        max_ep_len: int = 1000,
        unsuper_step: int = 0,
    ):

        super(OnPolicyRewardAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        
        # reward learning
        self.reward_model = reward_model
        self.thres_interaction = num_interaction
        self.num_feed = num_feed
        self.feed_type = feed_type
        self.re_update = re_update
        self.traj_obsact = None
        self.traj_reward = None
        self.first_reward_train = 0
        self.num_interactions = 0
        self.metaworld_flag = metaworld_flag
        self.max_feed = max_feed
        self.total_feed = 0
        self.labeled_feedback = 0
        self.noisy_feedback = 0
        if self.reward_model:
            self.reward_batch = self.reward_model.mb_size
        self.unsuper_step = unsuper_step
        self.avg_train_true_return = 0
        self.size_segment = size_segment
        self.max_ep_len = max_ep_len
        self.custom_logger = Logger(tensorboard_log, save_tb=False, log_frequency=10000, agent='ppo')

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        
        if self.unsuper_step > 0:
            self.unsuper_buffer = EntReplayBuffer(
                self.unsuper_step+100,
                self.observation_space,
                self.action_space,
                self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=False,
            )
            self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=self.device)
        
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
    
    def learn_reward(
        self) -> None:
        
        # update margin
        new_margin = np.mean(self.avg_train_true_return) * (self.size_segment / self.max_ep_len)
        self.reward_model.set_teacher_thres_skip(new_margin)
        self.reward_model.set_teacher_thres_equal(new_margin)
        
        if self.first_reward_train == 0:
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            else:
                raise NotImplementedError
        
        self.total_feed += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        # update reward
        for epoch in range(self.re_update):
            if self.reward_model.teacher_eps_equal > 0:
                train_acc = self.reward_model.train_soft_reward()
            else:
                train_acc = self.reward_model.train_reward()
            total_acc = np.mean(train_acc)
            
            if total_acc > 0.97:
                break;
                
        print("Reward function is updated!! ACC: " + str(total_acc))
    
    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            
            obsact = np.concatenate((self._last_obs, clipped_actions), axis=-1) # num_env x (obs+act)
            obsact = np.expand_dims(obsact, axis=1) # num_env x 1 x (obs+act) 
            
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            batch_reward = rewards.reshape(-1,1,1)
            
            pred_reward = self.reward_model.r_hat_batch(obsact)
            pred_reward = pred_reward.reshape(-1)
            
            if self.traj_obsact is None:
                self.traj_obsact = obsact
                self.traj_reward = batch_reward
            else:
                self.traj_obsact = np.concatenate((self.traj_obsact, obsact), axis=1)
                self.traj_reward = np.concatenate((self.traj_reward, batch_reward), axis=1)
            
            self.num_timesteps += env.num_envs
            self.num_interactions += env.num_envs
            
            # custome log
            num_dones = int(sum(dones))
            if num_dones > 0:
                # add samples to buffer
                self.reward_model.add_data_batch(self.traj_obsact, self.traj_reward)
                # reset traj
                self.traj_obsact, self.traj_reward = None, None
                
                # train reward using random data
                if self.first_reward_train == 0:
                    self.learn_reward()
                    self.first_reward_train = 1
                    self.num_interactions = 0
                else:
                    if self.num_interactions >= self.thres_interaction and self.total_feed < self.max_feed:
                        self.learn_reward()
                        self.num_interactions = 0
                
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    ep_reward = []
                    ep_success = []
                    for idx, info in enumerate(infos):
                        maybe_ep_info = info.get("episode")
                        if maybe_ep_info is not None:
                            ep_reward.append(maybe_ep_info["r"])
                            if self.metaworld_flag:
                                ep_success.append(maybe_ep_info["s"])
                                
                    self.custom_logger.log('eval/episode_reward', np.mean(ep_reward), self.num_timesteps)
                    self.custom_logger.log('eval/true_episode_reward', np.mean(ep_reward), self.num_timesteps)
                    if self.metaworld_flag:
                        self.custom_logger.log('eval/true_episode_success', np.mean(ep_success), self.num_timesteps)
                    self.custom_logger.dump(self.num_timesteps)

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, pred_reward, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    def collect_rollouts_unsuper(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, 
        n_rollout_steps: int, replay_buffer: EntReplayBuffer
    ) -> bool:

        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            
            obsact = np.concatenate((self._last_obs, clipped_actions), axis=-1) # num_env x (obs+act)
            obsact = np.expand_dims(obsact, axis=1) # num_env x 1 x (obs+act) 
            
            obs_origin = env.get_original_obs()
            replay_buffer.add_obs(obs_origin)
            state_entropy = replay_buffer.compute_state_entropy(obs_origin)
            self.s_ent_stats.update(state_entropy)
            norm_state_entropy = state_entropy / self.s_ent_stats.std

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            next_obs_origin = env.get_original_obs()            
            batch_reward = rewards.reshape(-1,1,1)
            
            pred_reward = norm_state_entropy.reshape(-1).data.cpu().numpy()
            
            if self.traj_obsact is None:
                self.traj_obsact = obsact
                self.traj_reward = batch_reward
            else:
                self.traj_obsact = np.concatenate((self.traj_obsact, obsact), axis=1)
                self.traj_reward = np.concatenate((self.traj_reward, batch_reward), axis=1)
            
            self.num_timesteps += env.num_envs
            self.num_interactions += env.num_envs
            
            # custome log
            num_dones = int(sum(dones))
            if num_dones > 0:
                # add samples to buffer
                if self.reward_model:
                    self.reward_model.add_data_batch(self.traj_obsact, self.traj_reward)
                # reset traj
                self.traj_obsact, self.traj_reward = None, None
                                
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    ep_reward = []
                    ep_success = []
                    for idx, info in enumerate(infos):
                        maybe_ep_info = info.get("episode")
                        if maybe_ep_info is not None:
                            ep_reward.append(maybe_ep_info["r"])
                            if self.metaworld_flag:
                                ep_success.append(maybe_ep_info["s"])
                                
                    self.custom_logger.log('eval/episode_reward', np.mean(ep_reward), self.num_timesteps)
                    self.custom_logger.log('eval/true_episode_reward', np.mean(ep_reward), self.num_timesteps)
                    if self.metaworld_flag:
                        self.custom_logger.log('eval/true_episode_success', np.mean(ep_success), self.num_timesteps)
                    self.custom_logger.dump(self.num_timesteps)

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, pred_reward, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    
    def collect_rollouts_true(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            
            # custome log
            num_dones = int(sum(dones))
            if num_dones > 0:
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    ep_reward, ep_success = [], []
                    for idx, info in enumerate(infos):
                        maybe_ep_info = info.get("episode")
                        if maybe_ep_info is not None:
                            ep_reward.append(maybe_ep_info["r"])
                            if self.metaworld_flag:
                                ep_success.append(maybe_ep_info["s"])

                    self.custom_logger.log('eval/episode_reward', np.mean(ep_reward), self.num_timesteps)
                    self.custom_logger.log('eval/true_episode_reward', np.mean(ep_reward), self.num_timesteps)
                    if self.metaworld_flag:
                        self.custom_logger.log('eval/true_episode_success', np.mean(ep_success), self.num_timesteps)
                    self.custom_logger.dump(self.num_timesteps)
                    
            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyRewardAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyRewardAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                logger.record("reward/total_feed", self.total_feed)
                logger.record("reward/labeled_feedback", self.labeled_feedback)
                logger.record("reward/noisy_feedback", self.noisy_feedback)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.avg_train_true_return = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                    logger.record("rollout/ep_rew_mean", self.avg_train_true_return)
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    if self.metaworld_flag:
                        logger.record("rollout/ep_success_mean", safe_mean([ep_info["s"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)
            
            if self.first_reward_train > 1:
                self.train()
                
            if self.first_reward_train == 1:
                self.first_reward_train += 1

        callback.on_training_end()

        return self
    
    def learn_unsuper(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyRewardAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyRewardAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())
 
        while self.num_timesteps < total_timesteps:
            if self.num_timesteps < self.unsuper_step:
                continue_training = self.collect_rollouts_unsuper(
                    self.env, callback, self.rollout_buffer, 
                    n_rollout_steps=self.n_steps, replay_buffer=self.unsuper_buffer)
            else:
                if self.first_reward_train == 0:
                    self.learn_reward()
                    self.num_interactions = 0
                    self.first_reward_train = 2
                    self.policy.reset_value()
                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
                                
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                logger.record("reward/total_feed", self.total_feed)
                logger.record("reward/labeled_feedback", self.labeled_feedback)
                logger.record("reward/noisy_feedback", self.noisy_feedback)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.avg_train_true_return = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                    logger.record("rollout/ep_rew_mean", self.avg_train_true_return)
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    if self.metaworld_flag:
                        logger.record("rollout/ep_success_mean", safe_mean([ep_info["s"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)
            
            
            if self.first_reward_train == 2:
                self.train()
            else:
                self.train_unsuper()

        callback.on_training_end()

        return self
    
    def learn_unsuper_true(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyRewardAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyRewardAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())
 
        while self.num_timesteps < total_timesteps:
            if self.num_timesteps < self.unsuper_step:
                continue_training = self.collect_rollouts_unsuper(
                    self.env, callback, self.rollout_buffer, 
                    n_rollout_steps=self.n_steps, replay_buffer=self.unsuper_buffer)
            else:
                if self.first_reward_train == 0:
                    self.first_reward_train = 2
                    self.policy.reset_value()
                continue_training = self.collect_rollouts_true(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
                                
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                logger.record("reward/total_feed", self.total_feed)
                logger.record("reward/labeled_feedback", self.labeled_feedback)
                logger.record("reward/noisy_feedback", self.noisy_feedback)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.avg_train_true_return = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                    logger.record("rollout/ep_rew_mean", self.avg_train_true_return)
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    if self.metaworld_flag:
                        logger.record("rollout/ep_success_mean", safe_mean([ep_info["s"] for ep_info in self.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)
            
            
            if self.first_reward_train == 2:
                self.train()
            else:
                self.train_unsuper()

        callback.on_training_end()

        return self
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []