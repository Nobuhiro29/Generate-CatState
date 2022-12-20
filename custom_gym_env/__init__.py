from gym.envs.registration import register

register(
    id = 'DoubleHarmonicOscillatorEnv-v0',
    entry_point = 'custom_gym_env.envs:DoubleHarmonicOscillatorEnv'
)