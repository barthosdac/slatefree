seeds=range(100)
N_iter=1000
N=50
K=10

episodes_per_iter=1
iter_per_test=1
onehot=True
warm_iters=25


print("debut")
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

from recsim_mod.simulator import runner
from recsim_mod.agents import slateq_agent
from recsim_mod.environments import interest_evolution

dir = './results/'

if not os.path.exists(dir) :
    os.mkdir(dir)
    
print("Starting")
for seed in seeds :
    seed_env=seed
    seed_agent=2**31-1-seed
    
    def create_slateq_agent(sess, environment, eval_mode, summary_writer=None):
        """
        This is one variant of the agent featured in SlateQ paper
        """
        kwargs = {
          'observation_space': environment.observation_space,
          'action_space': environment.action_space,
          'summary_writer': summary_writer,
          'eval_mode': False,
          'learn_state': False,
          'use_state': True,
          'implicit' : params['implicit'],
          'seed' : seed_agent,
          'approximateur' : 'u'
        }
        return slateq_agent.create_agent(agent_name="slate_greedy_greedy_q", sess=sess, **kwargs)

    env_config = {
        'num_candidates': N,
        'slate_size': K,
        'resample_documents': True,
        'seed': seed_env,
        'static' : True,
        'implicit' : params['implicit'],
        'onehot' : onehot,
        'warm_iters' : warm_iters
        }

    path=f"{dir}{seed}/"
    if not os.path.exists(path) :
        print(f"Starting seed : {seed}")
        runner_q = runner.Runner(
            create_agent_fn=create_slateq_agent,
            env=interest_evolution.create_environment(env_config),
            mono_utilisateur=False,
            episodes_per_iter=episodes_per_iter,
            iter_per_test=iter_per_test,
            base_dir=path,
            warm_iters=warm_iters)

        runner_q.run_experiment(N_iter)
        runner_q.save(logs=True,net=True)
