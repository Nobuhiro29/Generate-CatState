import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import csv
from qutip import *

from custom_gym_env.envs.double_harmonic_oscillator import *


class DoubleHarmonicOscillatorEnv(gym.Env):

    def __init__(self):
        self.min_action = -1
        self.max_action = 1
        self.min_measurement_current = -np.inf
        self.max_measurement_current = np.inf

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float
        )
        self.observation_space = spaces.Box(
            low=self.min_measurement_current, high=self.max_measurement_current, shape=(1,), dtype=np.float
        )
        self.oscillator = DoubleHarmonicOscillator()

        self.episode_count = -1

    def reset(self):
        self.episode_count += 1
        self.reward_list = []
        self.stepcount = 0

        os.makedirs('/media/user/3AB8C6C2B8C67C3F/Kasahara/result/episode{}'.format(self.episode_count),
                    exist_ok=True)
        
        self.rho0 = self.oscillator.init_state()

        return np.zeros(1)

    def step(self, action):
        # constant
        self.nsteps = 0.0025
        self.times = np.arange(0, 0.01, self.nsteps)
        self.ntraj = 8
        self.nsubsteps = 30

        # operators
        self.measurement_matrix = Qobj(self.oscillator.measurement_operator_matrix())
        self.sc_ops = [Qobj(self.oscillator.measurement_operator_matrix())]
        self.e_ops = [self.sc_ops[0]]
        self.H0 = self.oscillator.System_Hamiltonian()
        self.F = self.oscillator.squeezed_Hamiltonian()

        # action
        a0 = action[0] * 5

        def squeeze_coeff(t, args):
            return args["w1"]

        H = QobjEvo([self.H0, [self.F, squeeze_coeff]], args={"w1":a0})

        stoc_solution = smesolve(
            H, self.rho0, self.times, [], self.sc_ops, self.e_ops,
            ntraj=self.ntraj, nsubsteps=self.nsubsteps, method="homodyne",
            store_measurement=True, dW_factors=[1/np.sqrt(0.4)]
        )
        self.rho0 = stoc_solution.states[-1]
        
        with open('/media/user/3AB8C6C2B8C67C3F/Kasahara/result/episode{}/rho.csv'.format(self.episode_count), 'a') as f:
            writer_1 = csv.writer(f)
            writer_1.writerow(np.reshape(np.array(self.rho0), (-1,)))

        measurement_currents = [np.array(stoc_solution.measurement).mean(axis=0)[-4].real]
        amc = np.average(measurement_currents)

        self.stepcount += 1
        if self.stepcount == 1000:
            done = True
        else:
            done = False

        reward = -abs(amc - 3**2)

        with open('/media/user/3AB8C6C2B8C67C3F/Kasahara/result/episode{}/reward.csv'.format(self.episode_count), 'a') as f:
            writer_2 = csv.writer(f)
            writer_2.writerow(np.reshape(np.array(reward), (-1)))

        return np.array([amc]), reward, done, {}

    def render(self, mode='console', close=False):
        print('done')

    def close(self):
        pass