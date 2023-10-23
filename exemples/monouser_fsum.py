seeds=range(100)
N_iter=10000
N=50
K=10

episodes_per_iter=1
iter_per_test=1

implicit=True
warm_iters=None
use_state=False
onehot=True

min_replay_history=50
target_update_period=500

print("debut")
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

from recsim_mod.simulator import runner
from recsim_mod.agents import slatefree_agent
from recsim_mod.environments import interest_evolution


dir = './results/'

if not os.path.exists(dir) :
    os.mkdir(dir)


print("Starting")
for seed in seeds :
    seed_env=seed
    seed_agent=2**31-1-seed

    def create_slatefree_agent(sess, environment, eval_mode, summary_writer=None):
        """
        This is one variant of the agent featured in SlateQ paper
        """
        kwargs = {
        'observation_space': environment.observation_space,
        'action_space': environment.action_space,
        'summary_writer': summary_writer,
        'eval_mode': eval_mode,
        'min_replay_history': min_replay_history,
        'target_update_period': target_update_period,
        'learn_state': False,
        'use_state': use_state,
        'implicit' : implicit,
        'loss_calc' : "sum",
        'seed' : seed_agent
        }
        return slatefree_agent.create_agent(agent_name = "slate_topk_topk_q", sess=sess, **kwargs)

    env_config = {
    'num_candidates': N,
    'slate_size': K,
    'resample_documents': True,
    'seed': seed_env,
    'static' : True,
    'onehot' : onehot,
    'implicit' : implicit
    }


    #slateFree
    path=f"{dir}{seed}/"
    if not os.path.exists(path) :
        print(f"Starting seed : {seed}")
        runner_free = runner.Runner(
            create_agent_fn=create_slatefree_agent,
            env=interest_evolution.create_environment(env_config),
            mono_utilisateur=True,
            episodes_per_iter=episodes_per_iter,
            iter_per_test=iter_per_test,
            base_dir=path,
            warm_iters=warm_iters)
        runner_free.run_experiment(N_iter)
        runner_free.save()
