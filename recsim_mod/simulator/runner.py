# coding=utf-8
# coding=utf-8
# Copyright 2019 The RecSim Authors.
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
r"""An executable class to run agents in the simulator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags
import gin.tf
from gym import spaces
import numpy as np
from recsim_mod.simulator import environment
from recsim_mod import document
import tensorflow.compat.v1 as tf
import json
from copy import deepcopy

from math import log,floor
def tronque (x, cs=4) :
    """
    Truncate the float x keeping at leats cs significant figures.
    Never truncate number before the dot.
    """
    if x==0 :
        return x

    n=floor(log(abs(x))/log(10))
    if n>cs :
        return round(x,0)

    k=-n+cs
    return round(x,k)
    
    
def tronque_rec (o,cs=4):
    if isinstance(o,float) :
        return tronque(o,cs)
    elif isinstance(o,dict) :
        return {k: tronque_rec(v,cs) for k, v in o.items()}
    elif isinstance(o,list) :
        return [tronque_rec(k,cs) for k in o]
    else :
        return o

"""flags.DEFINE_bool(
    'debug_mode', False,
    'If set to true, the agent will output in-episode statistics '
    'to Tensorboard. Disabled by default as this results in '
    'slower training.')
flags.DEFINE_string('agent_name', None, 'Name of the agent.')
flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_string(
    'environment_name', 'interest_evolution',
    'The environment with which to run the experiment. Supported choices are '
    '{interest_evolution, interest_exploration}.')
flags.DEFINE_string(
    'episode_log_file', '',
    'Filename under base_dir to output simulated episodes in SequenceExample.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"third_party/py/dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "runner_lib.Runner.max_steps_per_episode=100')


FLAGS = flags.FLAGS"""


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in the
      config files.
  """
  gin.parse_config_files_and_bindings(
      gin_files, bindings=gin_bindings, skip_unknown=False)


@gin.configurable
class Runner(object):
  """Object that handles running experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.
  """
  _agent = None

  def __init__(self,
               create_agent_fn,
               env,
               base_dir,
               max_steps_per_episode=27000,
               episodes_per_iter=1,
               iter_per_test=1,
               mono_utilisateur=True,
               warm_iters=None):
    """Initializes the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      env: A Gym environment for running the experiments.
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
      mono_utilisateur: bool, si l'itération se fait sur un unique utilisateur
      inter_test: int, intervalle f=d'itération pour faire les tests
    """
    tf.logging.info('max_steps_per_episode = %s', max_steps_per_episode)

    self._create_agent_fn = create_agent_fn
    self._env = env
    self._max_steps_per_episode = max_steps_per_episode
    self._episodes_per_iter=episodes_per_iter
    self._iter_per_test=iter_per_test
    self._mono_utilisateur = mono_utilisateur
    self._base_dir=base_dir

    """Sets up the runner by creating and initializing the agent."""
    # Reset the tf default graph to avoid name collisions from previous runs
    # before doing anything else.
    tf.reset_default_graph()
    # Set up a session and initialize variables.
    self._sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    self._agent = self._create_agent_fn(
        self._sess,
        self._env,
        eval_mode=False)
    # type check: env/agent must both be multi- or single-user
    if self._agent.multi_user and not isinstance(
        self._env.environment, environment.MultiUserEnvironment):
      raise ValueError('Multi-user agent requires multi-user environment.')
    if not self._agent.multi_user and isinstance(
        self._env.environment, environment.MultiUserEnvironment):
      raise ValueError('Single-user agent requires single-user environment.')
    self._sess.run(tf.global_variables_initializer())
    self._sess.run(tf.local_variables_initializer())

    """Initializes the metrics."""
    self._reset_stats()
    # Initialize environment-specific metrics.
    self._env.reset_metrics()

    """Initialisation de train"""
    self.total_steps = 0
    self._log_test=[]
    self._warm_iters=warm_iters

  def _reset_stats(self) :
    self._stats = {
        'episode_length': [],
        'episode_time': [],
        'episode_reward': [],
        'last_obs_user' : [],
        'q_values' : [],
        'user_vector' : []
    }



  def _run_one_episode(self,observation):
    """Executes a full trajectory of the agent interacting with the environment.

    Takes :
      The initial observtion.
    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    start_time = time.time()
    sequence_example = tf.train.SequenceExample()

    #soft reset
    action = self._agent.begin_episode(observation)

    # Keep interacting until we reach a terminal state.
    while True :
      
      observation, reward, done, info = self._env.step(action)


      self._env.update_metrics(observation['response'], info)

      total_reward += reward
      step_number += 1


      if done:
        break
      elif step_number == self._max_steps_per_episode:
        # Stop the run loop once we reach the true end of episode.
        break
      else :
        action = self._agent.step(reward, observation)


    self._agent.end_episode(reward, observation)


    time_diff = time.time() - start_time
    self._update_episode_metrics(
        episode_length=step_number,
        episode_time=time_diff,
        episode_reward=total_reward,
        observation=observation)

    return step_number, total_reward

  def _update_episode_metrics(self, episode_length, episode_time,
                              episode_reward,observation):
    """Updates the episode metrics with one episode."""

    self._stats['episode_length'].append(episode_length)
    self._stats['episode_time'].append(episode_time)
    self._stats['episode_reward'].append(episode_reward)
    self._stats['last_obs_user'].append(list(observation['user']))
    self._stats['q_values'].append(list(self._agent._get_q_values()))
    try : 
      a, b = self._agent.user_vector.get_vecteur()
      self._stats['user_vector'].append([list(a),b])
    except : None

  def _warm_start (self) :
    eval_mode=self._agent.eval_mode
    warm_starting=self._agent._warm_starting

    self._agent.eval_mode = True #No training of the neural networks
    self._agent._warm_starting = True #No learning of the state

    if self._mono_utilisateur : #warm_iters as a number of iteration
      for _ in range(self._warm_iters) :
        self._run_eval_phase()
        self.total_steps = self._run_iteration(self.total_steps)
    else : #warm_iters as a number of episodes
      for _ in range(self._warm_iters) :
        observation = self._env.soft_reset()
        self.total_steps += self._run_one_episode(observation)[0]

    self._agent.eval_mode=eval_mode
    self._agent._warm_starting=warm_starting

  def run_experiment(self,num_iterations=101):
    """Runs a full experiment, spread over multiple iterations."""
    tf.logging.info('Beginning training...')
    

    if self._mono_utilisateur and self._warm_iters is not None:
      #if warm_start and only on user
      #it's the only learning of the user vector
      self._warm_start()
      num_iterations -= self._warm_iters

    self._agent.eval_mode=False

    for iteration in range(num_iterations):
      if iteration%10==0 :
        print('Starting iteration ', iteration)

      self._run_eval_phase()
      self.total_steps = self._run_iteration(self.total_steps)
    return True

  def _run_iteration(self, total_steps):
    """Runs training phase and updates total_steps.
    Run one iteraion"""

    num_steps = 0
    nb_episodes=self._episodes_per_iter
    if not self._mono_utilisateur :
      self._env.reset()
      if self._agent.user_vector is not None :
        self._agent.user_vector.reset()
      if self._warm_iters is not None :
        self._warm_start()
        nb_episodes-=self._warm_iters
    for k in range(nb_episodes) :
      observation = self._env.soft_reset()
      episode_length, _ = self._run_one_episode(observation)
      num_steps += episode_length

    total_steps += num_steps
    return total_steps

  def _run_eval_phase(self):
    """Runs evaluation phase given model has been trained for total_steps.
    Execute the max_iters iterations"""
    eval_mode=self._agent.eval_mode
    warm_starting=self._agent._warm_starting

    self._agent._warm_starting=False
    self._agent.eval_mode=True

    self._reset_stats()
    for k in range (self._iter_per_test) :
      self.total_steps = self._run_iteration(self.total_steps)

    self._log_test.append(deepcopy(self._stats))

    self._agent.eval_mode=eval_mode
    self._agent._warm_starting=warm_starting



  def save(self,logs=True,net=False,fname_log='log_test.json') :
    try : os.mkdir(self._base_dir)
    except : pass
    if logs : 
      with open(self._base_dir+fname_log, 'w+') as f:
          f.write(json.dumps(tronque_rec(self._log_test), indent = 0))
    if net :
      self._agent.save(self._base_dir)

  def load(self,net=True) :
    if net :
      self._agent.load(self._base_dir)