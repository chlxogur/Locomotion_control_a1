# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

class SimpleForwardTask(object):
  """Default empy task."""
  def __init__(self):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    self.past_motorvel = np.zeros(12)
    self.past_distance = 4.
    self.timesteps = 0
    self.feet_air_time = [0., 0., 0., 0.]
    self.last_contacts = [True, True, True, True]
    self.past_action = np.zeros(12)
    self.vel_limit = np.array([21., 21., 21., 21., 21., 21., 21., 21., 21., 21., 21., 21.])
    self.debug_txt_pos = 0
    self.debug_txt_tar = 0

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos
    self.past_distance = math.hypot(
            env.target_x - self.current_base_pos[0], env.target_y - self.current_base_pos[1])
    self.timesteps = 0
    self.feet_air_time = [0., 0., 0., 0.]
    self.last_contacts = [True, True, True, True]
    self.past_motorvel = env.robot.GetMotorVelocities()
    self.past_action = env._last_action

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()

  def done(self, env):
    """Checks if the episode is over.

       If the robot base becomes unstable (based on orientation), the episode
       terminates early.
    """
    self.timesteps = self.timesteps + 1

    current_base_position = env.robot.GetBasePosition()
    current_distance = math.hypot(
            env.target_x - current_base_position[0], env.target_y - current_base_position[1])
    
    roll, pitch, _ = env.robot.GetTrueBaseRollPitchYaw()
    orientation = env.robot.GetBaseOrientation()
    rot_mat = env._pybullet_client.getMatrixFromQuaternion(orientation)
    local_up = rot_mat[6:]
  
    fall_condb = math.fabs(roll) > 0.3 or math.fabs(pitch) > 0.5
    height_cond = current_base_position[2] < 0.2

    goal = current_distance < 0.5
    time = self.timesteps >= 2000
    
    '''
    if self.timesteps % 10 == 0:
        print('Current Position : ' + '(' + str(round(current_base_position[0], 2)) + ', ' + str(round(current_base_position[1], 2)) + '), ' + 'Target : ' + '('+ str(round(env.target_x, 2)) + ', ' + str(round(env.target_y, 2)) + ')')
    
    if goal == True:
        print('GOAL!!!')
    '''
    if goal == True:
        env.goal = True
        
    env._pybullet_client.removeUserDebugItem(self.debug_txt_pos)
    env._pybullet_client.removeUserDebugItem(self.debug_txt_tar)
    pos = 'POSITION : ' + str([round(current_base_position[0], 2), round(current_base_position[1], 2)])
    target = 'TARGET : ' + str([round(env.target_x, 2), round(env.target_y, 2)])

    self.debug_txt_pos = env._pybullet_client.addUserDebugText(pos, [current_base_position[0], current_base_position[1], 0.5], textColorRGB=[0,0,1], textSize=1)
    self.debug_txt_tar = env._pybullet_client.addUserDebugText(target, [current_base_position[0], current_base_position[1], 0.45], textColorRGB=[0,0,1], textSize=1)

    return height_cond

  def reward(self, env):
    """Get the reward without side effects."""
     
    current_base_position = env.robot.GetBasePosition()
    roll, pitch, yaw = env.robot.GetTrueBaseRollPitchYaw()
  
    current_distance = math.hypot(
            env.target_x - current_base_position[0], env.target_y - current_base_position[1])
    
    distance_rate = (self.past_distance - current_distance)
  
    if distance_rate > 0:
      reward = 2000.*distance_rate
    if distance_rate <= 0:
      reward = -8.

    contact = env.robot.GetFootContacts()
    contact_filt = [self.last_contacts[0] or contact[0], self.last_contacts[1] or contact[1], self.last_contacts[2] or contact[2], self.last_contacts[3] or contact[3]]
    self.last_contacts = contact
    first_contact = [(self.feet_air_time[0] > 0.) or contact_filt[0], (self.feet_air_time[1] > 0.) or contact_filt[1], (self.feet_air_time[2] > 0.) or contact_filt[2], (self.feet_air_time[3] > 0.) or contact_filt[3]]
   
    self.feet_air_time[0] += 0.001
    self.feet_air_time[1] += 0.001
    self.feet_air_time[2] += 0.001
    self.feet_air_time[3] += 0.001

    rew_airTime = 0.

    if first_contact[0]:
      rew_airTime += self.feet_air_time[0] - 0.5
    if first_contact[1]:
      rew_airTime += self.feet_air_time[1] - 0.5
    if first_contact[2]:
      rew_airTime += self.feet_air_time[2] - 0.5
    if first_contact[3]:
      rew_airTime += self.feet_air_time[3] - 0.5

    if contact_filt[0]:
      self.feet_air_time[0] = 0.
    if contact_filt[1]:
      self.feet_air_time[1] = 0.
    if contact_filt[2]:
      self.feet_air_time[2] = 0.
    if contact_filt[3]:
      self.feet_air_time[3] = 0.

    energy_reward = -np.abs(np.dot(env.robot.GetMotorTorques(),
                                   env.robot.GetMotorVelocities())) * 0.01 

    current_action = env._last_action
    action_rew = np.sum(np.square(current_action - self.past_action))
    action_rew = -action_rew * 0.1
    self.past_action = current_action

    current_motorvel = env.robot.GetMotorVelocities()
    motoracc_rew = np.sum(np.square((self.past_motorvel - current_motorvel) / 0.001))
    motoracc_rew = -motoracc_rew * 0.00000000001
  
    self.past_motorvel = current_motorvel

    motorvel_clip = np.clip(np.abs(current_motorvel) - self.vel_limit, 0., 5.)
    motorvel_rew = -np.sum(motorvel_clip)
   
    reward = reward + rew_airTime + energy_reward + motorvel_rew + motoracc_rew + action_rew

    #if self.timesteps % 10 == 0:
    #  print(action_rew)
    #  print(motoracc_rew)
    #  print('dist_reward : ' + str(reward))
    #  print('air_reward : ' + str(rew_airTime))  

    self.past_distance = current_distance

    fall_condb = math.fabs(roll) > 0.3 or math.fabs(pitch) > 0.5
    height_cond = current_base_position[2] < 0.2

    if current_distance < 0.5:
      reward = 1000.

    if fall_condb or height_cond:
      reward = -1000.
    
    return reward
    
    

    
