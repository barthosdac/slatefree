# slateq_agent.py
This is a modified copy of the file `slate_decomp_q_agent.py`.
## 0) Modification of Neural Networks
The creation of neural networks has been modified to be closer to the `dopamine` library.

Now, there is a `create_network` method to create the two neural networks, `online_convnet` and `target_convnet`. Then, three neural networks, `replay_net_outputs`, `net_outputs`, and `target_convnet`, are created using the first two convnets and the adapter.

A new network, `net_q_values`, is added with the sole purpose of evaluating the Q-value of the `online_convnet` for each document feature.
## 1) Addition of Variables in the `SlateQAgent` Class:
- `use_state` (bool): If the user state is added to the input of the neural network. Set to false in single-user cases (task 1 and 2) because the user state remains the same throughout the experiment.
- `learn_state` (bool): If the user state is learned simultaneously with the neural network.
- `implicit` (bool): If the item rewards for reinforcement learning are implicit. If yes, the reward is sought in the watchtime attribute; otherwise, it is sought in quality.
- `network_size` (int): Allows for modifying the size of the hidden layer in the neural network.
- `warm_starting` (bool): If the network is in warm-starting, meaning the user vector is learned without learning the neural network.
- `approximateur` (string): Indicates how the user vector is approximated. "None" in cases where there is no approximation; the vector $u$ is directly read. "u" in the case of interest $\tilde u$. "h" in the case of historic $h$.<br>
The approximation is done in the `step` method.
- `user_vector` (class): This variable contains the user approximation. It is set to "None" when there is no approximation, and the user is directly observable.
## 2) Addition of the `Interest_approx` Class:
This class learns the $\tilde u$ vector through maximum likelihood. In the case where $v(\bot)$ is fixed at 1, we have:
$$
\text{if }x\in\mathcal D\text{ : }
\nabla_{\tilde u}\log\mathcal{L}=x-\dfrac{\sum_{y\in A}ye^{\tilde uy}}{\sum_{y\in A}e^{\tilde uy}+e^1}
$$
$$
\text{if }x=\bot\text{ : }
\nabla_{\tilde u}\log\mathcal{L}=-\dfrac{\sum_{y\in A}ye^{\tilde uy}}{\sum_{y\in A}e^{\tilde uy}+e^1}
$$
Then, we juste have to do $\tilde u\leftarrow\tilde u+\varepsilon\nabla_{\tilde u}\log\mathcal L$ at each choice of item.

# slatefree_agent.py
This is a modified copy of the file `slateq_agent.py` with all the formulas adapted.
## 1) Addition of Variables in the `SlateFreeAgent` Class:
- `cond` (bool): If true, the Q-value is updated only if an item is clicked. If false, even if an item is not clicked, the Q-value is updated (with a reward of 0).
- `loss_calc` (string): Takes the values "sum", "mse" or "max" corresponding to the function used in the loss function.
