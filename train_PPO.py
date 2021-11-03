import gym
import argparse
import yaml
import os

from collections import OrderedDict
from stable_baselines3 import PPO_CUSTOM
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_dmcontrol_env, make_vec_metaworld_env
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from stable_baselines3.common.vec_env import VecNormalize

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dog_walk", help="environment ID")
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="logs/PPO/", type=str)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=123)
    parser.add_argument("--n-envs", help="# of parallel environments", type=int, default=16)
    parser.add_argument("--n-steps", help="# of steps to run for each environment per update", type=int, default=500)
    parser.add_argument("--lr", help="learning rate", type=float, default=3e-4)
    parser.add_argument("--total-timesteps", help="total timesteps", type=int, default=2000000)
    parser.add_argument("-b", "--batch-size", help="batch size", type=int, default=64)
    parser.add_argument("--ent-coef", help="coeff for entropy", type=float, default=0.0)
    parser.add_argument("--hidden-dim", help="dim of hidden features", type=int, default=1024)
    parser.add_argument("--num-layer", help="# of layers", type=int, default=2)
    parser.add_argument("--use-sde", help="Whether to use generalized State Dependent Exploration", type=int, default=1)
    parser.add_argument("--sde-freq", help="Sample a new noise matrix every n steps", type=int, default=4)
    parser.add_argument("--gae-lambda", help="Factor for trade-off of bias vs variance", type=float, default=0.92)
    parser.add_argument("--clip-init", help="Initial value of clipping", type=float, default=0.4)
    parser.add_argument("--n-epochs", help="Number of epoch when optimizing the surrogate loss", type=int, default=20)
    parser.add_argument("--normalize", help="Normalization", type=int, default=1)    
    args = parser.parse_args()
    
    metaworld_flag = False
    if 'metaworld' in args.env:
        metaworld_flag = True
    
    # log name
    env_name = args.env
        
    if args.normalize == 1:
        args.tensorboard_log += 'normalized_' + env_name + '/lr_'+str(args.lr)
    else:
        args.tensorboard_log += env_name + '/lr_'+str(args.lr)
        
    args.tensorboard_log += '_batch_' + str(args.batch_size)
    args.tensorboard_log += '_nenvs_' + str(args.n_envs)
    args.tensorboard_log += '_nsteps_' + str(args.n_steps)
    args.tensorboard_log += '_ent_' + str(args.ent_coef)
    args.tensorboard_log += '_hidden_' + str(args.hidden_dim)
    args.tensorboard_log += '_sde_' + str(args.use_sde)
    args.tensorboard_log += '_sdefreq_' + str(args.sde_freq)
    args.tensorboard_log += '_gae_' + str(args.gae_lambda)
    args.tensorboard_log += '_clip_' + str(args.clip_init)
    args.tensorboard_log += '_nepochs_' + str(args.n_epochs)
    args.tensorboard_log += '_seed_' + str(args.seed) 
    
    # extra params
    if args.use_sde == 0:
        use_sde = False
    else:
        use_sde = True
    
    clip_range = linear_schedule(args.clip_init)
    
    # Parallel environments
    if metaworld_flag:
        env = make_vec_metaworld_env(
            args.env, 
            n_envs=args.n_envs, 
            monitor_dir=args.tensorboard_log,
            seed=args.seed)
    else:
        env = make_vec_dmcontrol_env(    
            args.env, 
            n_envs=args.n_envs, 
            monitor_dir=args.tensorboard_log,
            seed=args.seed)
    
    if args.normalize == 1:
        env = VecNormalize(env, norm_reward=False)
    
    # network arch
    net_arch = [dict(pi=[args.hidden_dim]*args.num_layer, 
                     vf=[args.hidden_dim]*args.num_layer)]
    policy_kwargs = dict(net_arch=net_arch)
    
    # train model
    model = PPO_CUSTOM(
        MlpPolicy, env,
        tensorboard_log=args.tensorboard_log, 
        seed=args.seed, 
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        use_sde=use_sde,
        sde_sample_freq=args.sde_freq,
        gae_lambda=args.gae_lambda,
        clip_range=clip_range,
        n_epochs=args.n_epochs,
        metaworld_flag=metaworld_flag,
        verbose=1)

    # save args
    with open(os.path.join(args.tensorboard_log, "args.yml"), "w") as f:
        ordered_args = OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
        yaml.dump(ordered_args, f)
    
    model.learn(total_timesteps=args.total_timesteps)