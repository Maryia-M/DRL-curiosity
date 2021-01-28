import argparse
import importlib
import os
import subprocess
from dotmap import DotMap
import gym
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common import logger
from stable_baselines3.reachability import CuriosityEnvWrapper
from stable_baselines3.reachability import RNetwork
from stable_baselines3.reachability.episodic_memory import EpisodicMemory
from stable_baselines3.reachability.rnet_trainer import RNetworkTrainer

import utils

# Runner with episodic curiosity

def train_and_test_ec(config, video_length_ = 1000, total_timesteps_ = 10000):
    print(config)
    if config.atari_wrapper:
        train_env = make_atari_env(config.environment, n_envs=config.workers)
        train_env = VecFrameStack(train_env, n_stack = 1)
        shape = (84, 84, 1)
    else:
        train_env = make_vec_env(config.environment, n_envs=config.workers)
        shape = train_env.observation_space.shape

    rnet = RNetwork(shape, config.ensemble_size)
    vec_episodic_memory = [EpisodicMemory([64], rnet.embedding_similarity, replacement='random', capacity=200)
                            for _ in range(config.workers)]
    target_image_shape = list(shape)
    #assert type(config.add_stoch) == bool, "Please indicated whether or not you want stoch added"
    train_env =  CuriosityEnvWrapper(train_env, vec_episodic_memory, rnet.embed_observation, target_image_shape, config.add_stoch)
    r_network_trainer = RNetworkTrainer(rnet, learning_rate=config.rnet_lr, observation_history_size=2000, training_interval=1000)
    train_env.add_observer(r_network_trainer)
    tb_dir = os.path.join(config.log_dir, config.tb_subdir)
    model = config.agent(config.policy_model, train_env, config, verbose=config.verbose, tensorboard_log=tb_dir)

    model.learn(total_timesteps=total_timesteps_)

    print("stopped to learn")
    #model.save("models/"+config.experiment)

    obs = train_env.reset()

    for i in range(video_length_+1):

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = train_env.step(action)
        train_env.render()
        if done.any():
            obs = train_env.reset()

    train_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="episodic-curiosity")
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

    train_and_test_ec(config)