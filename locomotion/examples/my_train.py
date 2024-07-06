"""Simple script for executing random actions on A1 robot."""
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from absl import app
from absl import flags
from tqdm import tqdm

from locomotion.envs import env_builder
from locomotion.robots import a1
from locomotion.robots import laikago
from locomotion.robots import robot_config
from locomotion.learning import imitation_policies as imitation_policies
from stable_baselines import PPO1
from stable_baselines.common.callbacks import CheckpointCallback

FLAGS = flags.FLAGS
flags.DEFINE_enum('robot_type', 'A1', ['A1', 'Laikago'], 'Robot Type.')
flags.DEFINE_enum('motor_control_mode', 'Torque',
                  ['Torque', 'Position', 'Hybrid'], 'Motor Control Mode.')
flags.DEFINE_bool('on_rack', False, 'Whether to put the robot on rack.')

ROBOT_CLASS_MAP = {'A1': a1.A1, 'Laikago': laikago.Laikago}

MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}
def main(_):
  robot = ROBOT_CLASS_MAP[FLAGS.robot_type]
  motor_control_mode = MOTOR_CONTROL_MODE_MAP['Position']
  #env = env_builder.build_regular_env(robot,
  #                                    motor_control_mode=motor_control_mode,
  #                                    enable_rendering=False,
  #                                    on_rack=FLAGS.on_rack)
  #env = gym.make('locomotion:A1GymEnv-v1')

  #env = make_vec_env('locomotion:A1GymEnv-v1', n_envs=4)
  env = gym.make('locomotion:A1GymEnv-v1')

  policy_kwargs = {
      "net_arch": [{"pi": [512, 256],
                    "vf": [512, 256]}],
      "act_fun": tf.nn.relu
      
  }

  checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path='./logs/', name_prefix='a1_cp')
    
  model = PPO1(policy=imitation_policies.ImitationPolicy,
               env=env,
               gamma=0.95,
               timesteps_per_actorbatch=4096,
               clip_param=0.2,
               optim_epochs=1,
               optim_stepsize=1e-5,
               optim_batchsize=256,
               lam=0.95,
               adam_epsilon=1e-5,
               schedule='constant',
               policy_kwargs=policy_kwargs,
               verbose=1,
               tensorboard_log="./a1_tensorboard/")
               
  model.learn(total_timesteps=17000000, callback=checkpoint_callback)
  model.save("ppo_a1")

  
#  del model

#  model = PPO.load("ppo_a1")

#  obs = env.reset()
#  for _ in tqdm(range(1000)):
#    action, state = model.predict(obs)
#    obs, reward, done, info = env.step(action)
    
if __name__ == "__main__":
  app.run(main)
