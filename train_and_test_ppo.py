import argparse
import importlib
import os
import subprocess
from dotmap import DotMap
import gym
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common import logger
from stable_baselines3.dummy import DummyEnvWrapper
from stable_baselines3.common.vec_env import VecVideoRecorder
import numpy as np
import imageio

import utils

# Regular Runner, no episodic curiosity

def create_environment(config):
    if config.atari_wrapper:
        env = make_atari_env(config.environment, n_envs=config.workers)
        env = VecFrameStack(env, n_stack = 1)
    else:
        env = make_vec_env(config.environment, n_envs=config.workers)
    env = DummyEnvWrapper(env, config.add_stoch)
    return env

def train_and_test_ppo(config, video_length_ = 1000, total_timesteps_ = 10000):
    print(config)
    train_env = create_environment(config)
    
    print("created training environment")
    tb_dir = os.path.join(config.log_dir, config.tb_subdir)
    model = config.agent(config.policy_model, train_env, config, verbose=config.verbose, tensorboard_log=tb_dir)
    print("started to learn")
    model.learn(total_timesteps=total_timesteps_)
    
    print("stopped to learn")
    #model.save("models/"+config.experiment)

    env = create_environment(config)
    obs = env.reset()

    for i in range(video_length_+1):
        #images.append(img)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done.any():
            obs = env.reset()
        
    env.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="drl_curiosity")
    parser.add_argument("-exp", "--experiment", type=str, required=True, help='name of config file in experiment folder')
    parser.add_argument("--log_dir", type=str, default='logs')
    parser.add_argument('--tb_port', action="store", type=int, default=6006, help="tensorboard port")

    # per run args
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--final_vis", type=bool, default=True)
    parser.add_argument("--final_vis_steps", type=int, default=1000)

    args = parser.parse_args()
    config = importlib.import_module('experiments.{}'.format(args.experiment)).config

    # update config dotmap
    config = config.toDict()
    config.update(vars(args))
    config = DotMap(config)

    # kill existing tensorboard processes on port (in order to refresh)
    utils.kill_processes_on_port(config.tb_port)

    env = dict(os.environ)   # Make a copy of the current environment
    subprocess.Popen('tensorboard --logdir ./{}'.format(config.log_dir), env=env, shell=True)


    train_and_test_ppo(config)