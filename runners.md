# runner.py
This is an adapted version of `runner_lib.py`.

It has the following parameters:
- `max_steps_per_episode` (int): The maximum number of steps per episode.
- `episodes_per_iter` (int): The number of episodes per iteration.
- `iter_per_test` (int): The number of test iterations after each training iteration.
- `mono_utilisateur` (bool): If true, the same user is retained across different iterations.
- `warm_iters` (int): If `mono_utilisateur` is true, it corresponds to the number of warm-starting iterations. The `warm_start` method will be called by `run_iteration`.<br>
 If `mono_utilisateur` is false, it corresponds to the number of warm-starting episodes in each iteration. The `warm_start` method will be called by `run_iteration`.

The `run_experiment` method will simulate a certain number of training episodes. Between each training episode, there will be test episodes to measure: the length of an episode, the total reward of an episode, the last user observation, the list of Q-values from the neural network, and the user approximation if it exists.

All measurements are recorded in the `log_test` variable. Unlike `runner_lib.py`, the measurements are stored in JSON files rather than TensorFlow files.

# runner_offline.py
A modified copy of `runner.py`.

This runner allows for learning from an `env_offline` environment.
