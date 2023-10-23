# interest_evolution.py

## 1) Additions of Variables:
- `implicit` (bool): If true, an item's reward corresponds to its watchtime, and user watchtime is updated based on the item's quality, varying the episode length. If false, the episode length is fixed, and an item's reward corresponds to its quality.

- `static` (bool): If true, a user has a fixed vector $u$. If false, it varies over time.

- `onehot` (bool): If true, an item is represented by a one-hot vector. Otherwise, it is represented by a continuous feature vector between 0 and 1.

## 2) Addition of Methods:
- `soft_reset` in `IEvUserModel`: Allows resetting a user's watchtime without resetting their vector $u$ (useful in single-user scenarios). This function is also added to `environment.py`.

- `sample_documents` in `IEvVideoSampler` and `UtilityModelVideoSampler`: Allows document generation in this file rather than in `environment.py`.

# simple_user.py
A modified copy of `interest_evolution.py`. Here, a vector is no longer modeled as $u$ but as a set $\mathcal Y$ of features. There are three choice functions:
- `user1`: The user has a probability $\alpha$ of not choosing an item. There is a probability of $1-\alpha$ to randomly choose an item uniformly from the slate.

- `user2`: The user chooses an item randomly from the slate only if all items do not have features in $\mathcal Y$.

- `user3`: The user chooses an item randomly from the slate only if at least one item has a feature in $\mathcal Y$.

# env_offline.py
A modified version of `interest_evolution.py` for data analysis. The data should be in the form of the following files:

- `item_map.json`: A dictionary $itemid\rightarrow itemk$. It links the item's ID to its position in the embedding matrix.

- `q.npy`: A matrix $itemk\rightarrow latent\ vector$. It provides the embedding of an item.

- `categories.json`: A dictionary $itemid\rightarrow category$. It gives the category of an item.

- `mean_categories.json`: A dictionary $category\rightarrow latent\ vector$.

- `sessions.json`: A dictionary $userid\rightarrow [[click^+],[(buy,quantity)^+]?]$. It represents the episodes in the data, where the $clicks$ and $buy$ are itemids.

The `Embendding` class is then created to load these files from their locations. It includes the `vecteur_of` method, which, given an item's ID, returns its embedding, the average of its category if its embedding is unknown, or a zero vector if its category is unknown.

1) First Training Method: `implicit = False`, $N>K$

For each session, an episode is considered as the sequence of different user clicks $c_t$. The agent is successively offered $N$ documents randomly, from which they choose $K$ to compose a slate. We simulate the user choosing the item closest to the session's document in the latent space.

The reward comes from the document quality and is calculated by:

2) Second Training Method: `implicit = True`, $N=K$

Similar to the simulation, here a fixed reward is taken. Each slate is chosen randomly, with the only constraint that it contains the clicked item chosen by the user. Here, we have off-policy learning because the slate selection action is not decided by the agent.

In both methods, we ensure that there is no Q-value update at the end of the episode when the user does not choose an item (which is not the default behavior in slateFree). We do not assume that a non-choice corresponds to a bad slate.
