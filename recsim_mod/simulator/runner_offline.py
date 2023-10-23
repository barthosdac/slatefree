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
import tensorflow.compat.v1 as tf
import json

from math import log,floor
def tronque (x, cs=4) :
    """
    tronque le float x en gardant au moins cs chiffres significatifs
    ne tronque jamais les nombres avant la virgule

    Utilisé pour que les fichiers json de log soient plus léger
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
               base_dir):
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
    self._create_agent_fn = create_agent_fn
    self._env = env
    self._base_dir=base_dir

    """Sets up the runner by creating and initializing the agent."""
    # Reset the tf default graph to avoid name collisions from previous runs
    # before doing anything else.
    tf.reset_default_graph()
    # Set up a session and initialize variables.
    self._sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    self._agent = self._create_agent_fn(
        self._sess,
        observation_space=env.observation_space,
        action_space=env.action_space)
    # type check: env/agent must both be multi- or single-user
    if self._agent.multi_user and not isinstance(
        self._env.environment, environment.MultiUserEnvironment):
      raise ValueError('Multi-user agent requires multi-user environment.')
    if not self._agent.multi_user and isinstance(
        self._env.environment, environment.MultiUserEnvironment):
      raise ValueError('Single-user agent requires single-user environment.')
    self._sess.run(tf.global_variables_initializer())
    self._sess.run(tf.local_variables_initializer())

    # Initialize environment-specific metrics.
    self._env.reset_metrics()

    """Initialisation de train"""
    self.total_steps = 0
    self._log_test={'recall':[]}



  def _run_one_episode(self,session):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.


    #soft reset
    self._env._environment._document_sampler.load_session(session)
    self._env._environment._document_sampler.eval_mode = self._agent.eval_mode
    observation=self._env.reset()
    self._env._environment._user_model.load_session(session)
    self._agent.user_vector.reset()

    click_ids=self._env._environment._document_sampler._clicks

    action = self._agent.begin_episode(observation)

    # Keep interacting until we reach a terminal state.
    while True :

      if self._agent.eval_mode and step_number<len(click_ids):
        docs_id=list(observation['doc'].keys())
        slate_docs_id=[docs_id[k] for k in action]
        doc=str(click_ids[step_number])
        print("slate_docs : ",slate_docs_id)
        print("doc : ",doc)
        self._log_test['recall'].append(int(doc in slate_docs_id))

      self._env._environment._document_sampler._iter+=1
      self._env._environment._user_model._user_state._iter+=1


      observation, reward, done, info = self._env.step(action)

      self._env.update_metrics(observation['response'], info)

      total_reward += reward
      step_number += 1

      if done:
        break
      else :
        action = self._agent.step(reward, observation)
      
    self._agent.end_episode(reward, observation)



    return step_number, total_reward

  def run_training(self,fname_sessions_train):
    """Runs a full experiment, spread over multiple iterations."""

    with open(fname_sessions_train, 'r') as f:
        sessions=json.load(f)


    print("\n*Beginning Training*\n")
    self._agent.eval_mode=False
    N=len(sessions.items())
    p=0
    for k,(id,session) in enumerate(sessions.items()):
      if 100*k/N>=p+0.1 :
         p=100*k/N
         print(f'Starting {k}th session : {p:0.2f}%')
         

      self._run_one_episode(session)
    return True

  def run_testing(self,fname_sessions_test):
    with open(fname_sessions_test, 'r') as f:
        sessions=json.load(f)
    print("\n*Beginning Testing*\n")
    self._agent.eval_mode=True
    N=len(sessions.items())
    p=0
    for k,(id,session) in enumerate(sessions.items()):
      if 100*k/N>=p+1 :
         p=100*k/N
         print(f'Starting {k}th session : {p:0.1f}%')
         

      self._run_one_episode(session)
    return True

  def save(self) :
    try : os.mkdir(self._base_dir)
    except : pass

    self._agent.bundle_and_checkpoint(self._base_dir,0)
    with open(self._base_dir+'recall.json', 'w+') as f:
        f.write(json.dumps(self._log_test, indent = 0))
  def load(self) :
    self._agent.unbundle(self._base_dir,0,None)