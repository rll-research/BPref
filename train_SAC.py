#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import utils
import hydra

from logger import Logger
from replay_buffer import ReplayBuffer

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)

        self.device = torch.device(cfg.device)
        self.log_success = False
        self.step = 0
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        
        # no relabel
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        meta_file = os.path.join(self.work_dir, 'metadata.pkl')
        pkl.dump({'cfg': self.cfg}, open(meta_file, "wb"))

    def evaluate(self):
        average_episode_reward = 0
        if self.log_success:
            success_rate = 0
            
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)
                episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            average_episode_reward += episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                        self.step)
        self.logger.dump(self.step)
        
    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        start_time = time.time()
        fixed_start_time = time.time()
        
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    self.logger.log('train/total_duration',
                                    time.time() - fixed_start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                            
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update             
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps) and self.cfg.num_unsup_steps > 0:
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
            
            
            next_obs, reward, done, extra = self.env.step(action)      
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            self.replay_buffer.add(
                obs, action, 
                reward, next_obs, done,
                done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

        self.agent.save(self.work_dir, self.step)
        
@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
