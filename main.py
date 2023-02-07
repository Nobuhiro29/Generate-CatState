import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import torch as th
from custom_gym_env.envs.double_quantum_environment import DoubleHarmonicOscillatorEnv
from custom_gym_env.envs.double_harmonic_oscillator import DoubleHarmonicOscillator

def make_env(env_id, rank, seed=0):

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_id = "DoubleHarmonicOscillatorEnv-v0"
    num_cpu = 8
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    
    policy_kwargs = dict(net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])])
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-5, batch_size=100, n_steps=1000)

    # Train the model
    model.learn(total_timesteps=1000000)

    env.close()
