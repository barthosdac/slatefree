"""Agent that implements the Slate-Q algorithms."""
import gin.tf
import numpy as np
from recsim_mod import agent as abstract_agent
from recsim_mod import choice_model
from recsim_mod.agents.dopamine import dqn_agent
import tensorflow.compat.v1 as tf
from copy import deepcopy
class Historic (object) :
    def __init__(self,nb_features,gamma=1) :
        self.num_features=nb_features
        self.historic_topics=np.zeros(nb_features)
        self.historic_papri=0
        self.step=0
        self.gamma=gamma

    def train(self,topic_slate,item_click) :
        N=len(topic_slate)
        exp=np.exp(topic_slate@self.historic_topics)
        if item_click>=0 :
          c=topic_slate[item_click]-np.delete(topic_slate,item_click,0).mean(axis=0)
          papri=0
        else :
          c=-topic_slate.mean(axis=0)
          papri=1

        if self.gamma==1 :
          self.historic_topics=self.historic_topics*self.step/(self.step+1)+c/(self.step+1)
          self.historic_papri=self.historic_papri*self.step/(self.step+1)+papri/(self.step+1)
        else :
          self.historic_topics=self.historic_topics*self.gamma*(1-self.gamma**self.step)/(1-self.gamma**(self.step+1)) + \
            c*(1-self.gamma)/(1-self.gamma**(self.step+1))
          self.historic_topics=self.historic_papri*self.gamma*(1-self.gamma**self.step)/(1-self.gamma**(self.step+1)) + \
            papri*(1-self.gamma)/(1-self.gamma**(self.step+1))
        self.step+=1

    def train_from_row (self,slate,docs,responses) :
        """
        train the historic from raw data
        """
        if type(slate) == type(None) : return None

        item_click=-1
        for i in range(len(responses)) :
          if responses[i]['click'] :
            item_click=i
            break

        topics_docs=[item[1] for item in list(docs.items())]
        topics_slates=np.array([topics_docs[i] for i in slate])
        self.train(topics_slates,item_click)

    def reset(self) :
      self.historic_topics=np.zeros(self.num_features)
      self.historic_papri=0
      self.step=0

    def get_vecteur (self) :
        """
        Return interest in each topic as a list
        """
        return list(self.historic_topics), self.historic_papri

    def save (self,dir) :
        """
        Save historic in a file
        """
        with open(dir+'historic', 'w+') as f :

          f.write(f"{self.historic_papri}\n")
          for i in self.historic_topics :
            f.write(f"{i}\n")
    def load (self,dir) :
        """
        Load historic in a file
        """
        with open(dir+'interest', 'r') as f :
          l=f.readlines()
          self.historic_papri=int(l[0][:-1])
          self.historic_topics=np.array([float(k[:-1]) for k in l[1:]])


class Interest_approx (object) :
    def __init__(self,nb_features,epsilon=0.01,random=False) :
        self.num_features=nb_features
        self.epsilon=epsilon
        self.random=random
        if random :
          self.interest_topics=2*np.random.random(nb_features)-1
        else : 
          self.interest_topics=np.zeros(nb_features)
        self.interest_noclick=1

    def train(self,topic_slate,item_click) :
        N=len(topic_slate)
        exp=np.exp(topic_slate@self.interest_topics)
        if item_click>=0 :
          
          grad_topic=topic_slate[item_click]-(topic_slate*exp[:,None]).sum(axis=0)/(exp.sum()+np.exp(self.interest_noclick))
        else :
          grad_topic=-(topic_slate*exp[:,None]).sum(axis=0)/(exp.sum()+np.exp(self.interest_noclick))

        self.interest_topics+=self.epsilon*grad_topic

    def reset(self) :
      if self.random :
        self.interest_topics=2*np.random.random(self.num_features)-1
      else :
        self.interest_topics=np.zeros(self.num_features)
      self.interest_noclick=1

    def train_from_row (self,slate,docs,responses) :
        """
        Train the interest from raw data
        """
        if type(slate) == type(None) : return None

        item_click=-1
        for i in range(len(responses)) :
          if responses[i]['click'] :
            item_click=i
            break

        topics_docs=[item[1] for item in list(docs.items())]
        topics_slates=np.array([topics_docs[i] for i in slate])
        self.train(topics_slates,item_click)

    def get_vecteur (self) :
        """
        Return interest in each topic as a list
        """
        if type(self.interest_topics)==type(None) :
          return self.interest_topics, self.interest_noclick
        else :
          return list(self.interest_topics), self.interest_noclick

    def save (self,dir) :
        """
        Save historic in a file
        """
        with open(dir+'interest', 'w+') as f :

          f.write(f"{self.interest_noclick}\n")
          for i in self.interest_topics :
            f.write(f"{i}\n")
    def load (self,dir) :
        """
        Load historic in a file
        """
        with open(dir+'interest', 'r') as f :
          l=f.readlines()
          self.interest_noclick=int(l[0][:-1])
          self.interest_topic=np.array([float(k[:-1]) for k in l[1:]])
def select_slate_topk(slate_size, q):
  """Selects the slate using the top-K algorithm.

  This algorithm corresponds to the method "TS" in
  Ie et al. https://arxiv.org/abs/1905.12767.

  Args:
    slate_size: int, the size of the recommendation slate.
    q: [num_of_documents] tensor, the predicted q values for documents.

  Returns:
    [slate_size] tensor, the selected slate.
  """
  _, output_slate = tf.math.top_k(q, k=slate_size)
  return output_slate

def compute_target_sarsa(reward, gamma, next_actions, next_q_values,
                         next_states, terminals):
  """Computes the SARSA target Q value.

  Args:
    reward: [batch_size] tensor, the immediate reward.
    gamma: float, discount factor with the usual RL meaning.
    next_actions: [batch_size, slate_size] tensor, the next slate.
    next_q_values: [batch_size, num_of_documents] tensor, the q values of the
      documents in the next step.
    next_states: [batch_size, 1 + num_of_documents] tensor, the features for the
      user and the docuemnts in the next step.
    terminals: [batch_size] tensor, indicating if this is a terminal step.

  Returns:
    [batch_size] tensor, the target q values.
  """
  stack_number = -1

  batch_size = next_q_values.get_shape().as_list()[0]
  slate_size = next_actions.get_shape().as_list()[1]
  print(f"K = {slate_size}")
  next_sarsa_q_list = []
  for i in range(batch_size):
    q = next_q_values[i]

    slate = tf.expand_dims(next_actions[i], 1)
    q_selected = tf.gather(q, slate)
    next_sarsa_q_list.append(
          tf.reduce_sum(input_tensor=q_selected)/(slate_size+1))

  next_sarsa_q_values = tf.stack(next_sarsa_q_list)


  return reward + gamma * next_sarsa_q_values * (1. - tf.cast(terminals, tf.float32))

def compute_target_topk_q(reward, gamma, next_actions, next_q_values,
                          next_states, terminals):
  """Computes the optimal target Q value with the greedy algorithm.

  This algorithm corresponds to the method "TT" in
  Ie et al. https://arxiv.org/abs/1905.12767.

  Args:
    reward: [batch_size] tensor, the immediate reward.
    gamma: float, discount factor with the usual RL meaning.
    next_actions: [batch_size, slate_size] tensor, the next slate.
    next_q_values: [batch_size, num_of_documents] tensor, the q values of the
      documents in the next step.
    next_states: [batch_size, 1 + num_of_documents] tensor, the features for the
      user and the docuemnts in the next step.
    terminals: [batch_size] tensor, indicating if this is a terminal step.

  Returns:
    [batch_size] tensor, the target q values.
  """
  batch_size = next_q_values.get_shape().as_list()[0]
  slate_size = next_actions.get_shape().as_list()[1]
  print(f"K = {slate_size}")
  next_topk_q_list = []
  for i in range(batch_size):
    q = next_q_values[i]

    slate = select_slate_topk(slate_size, q)
    q_selected = tf.gather(q, slate)
    next_topk_q_list.append(
          tf.reduce_sum(input_tensor=q_selected)/(slate_size+1))

  next_topk_q_values = tf.stack(next_topk_q_list)

  return reward + gamma * next_topk_q_values * (1. - tf.cast(terminals, tf.float32))

@gin.configurable
class SlateFreeAgent(dqn_agent.DQNAgentRecSim,
                        abstract_agent.AbstractEpisodicRecommenderAgent):
  """A recommender agent implements DQN using slate decomposition techniques."""

  def __init__(self,
               sess,
               observation_space,
               action_space,
               optimizer_name='',
               select_slate_fn=None,
               compute_target_fn=None,
               stack_size=1,
               network_size=256,
               eval_mode=False,
               use_state=True,
               learn_state=False,
               implicit=True,
               cond=False,
               loss_calc='mse',
               approximateur='u',
               seed=None,
               **kwargs):
    """Initializes SlateFreeAgent.

    Args:
      sess: a Tensorflow session.
      observation_space: A gym.spaces object that specifies the format of
        observations.
      action_space: A gym.spaces object that specifies the format of actions.
      optimizer_name: The name of the optimizer.
      select_slate_fn: A function that selects the slate.
      compute_target_fn: A function that omputes the target q value.
      stack_size: The stack size for the replay buffer.
      eval_mode: A bool for whether the agent is in training or evaluation mode.
      **kwargs: Keyword arguments to the DQNAgent.
    """

    if seed is not None :
      np.random.seed(seed)
      tf.set_random_seed(seed)

    self._response_adapter = dqn_agent.ResponseAdapter(
        observation_space.spaces['response'])
    response_names = self._response_adapter.response_names
    expected_response_names = ['click', 'watch_time']
    if not all(key in response_names for key in expected_response_names):
      raise ValueError(
          "Couldn't find all fields needed for the decomposition: %r" %
          expected_response_names)

    self._click_response_index = response_names.index('click')
    if implicit : 
      self._reward_response_index = response_names.index('watch_time')
    else : 
      self._reward_response_index = response_names.index('quality')
    self._quality_response_index = response_names.index('quality')
    self._observation_spacecluster_id_response_index = response_names.index('cluster_id')

    self._env_action_space = action_space
    self._num_candidates = int(action_space.nvec[0])
    abstract_agent.AbstractEpisodicRecommenderAgent.__init__(self, action_space)

    # The doc score is a [num_candidates] vector.
    ##self._doc_affinity_scores_ph = tf.placeholder(
    ##    tf.float32, (self._num_candidates,), name='doc_affinity_scores_ph')
    ##self._prob_no_click_ph = tf.placeholder(
    ##    tf.float32, (), name='prob_no_click_ph')

    self._select_slate_fn = select_slate_fn
    self._compute_target_fn = compute_target_fn

    nb_features=observation_space['doc']['0']._shape[0]
    if approximateur is None :
      self.user_vector = None
    elif approximateur == 'u' :
      self.user_vector=Interest_approx(nb_features=nb_features,random=not learn_state)
    elif approximateur == 'h' :
      self.user_vector=Historic(nb_features=nb_features)
        
    self._last_action=None
    self._use_state=use_state
    self._learn_state=learn_state
    self._warm_starting = False
    self._cond=cond
    self._loss_calc=loss_calc

    self._network_size=network_size

    self._user_ph = tf.placeholder(
        tf.float32, (nb_features)) #For Q evaluation

    dqn_agent.DQNAgentRecSim.__init__(
        self,
        sess,
        observation_space,
        num_actions=0,  # Unused.
        stack_size=1,
        optimizer_name=optimizer_name,
        eval_mode=eval_mode,
        **kwargs)

  def _network_adapter(self, states, net):
    self._validate_states(states)
      # Since we decompose the slate optimization into an item-level
      # optimization problem, the observation space is the user state
      # observation plus all documents' observations. In the Dopamine DQN agent
      # implementation, there is one head for each possible action value, which
      # is designed for computing the argmax operation in the action space.
      # In our implementation, we generate one output for each document.
    q_value_list = []
    for i in range(self._num_candidates):
      doc = tf.squeeze(states[:, i + 1, :, :], axis=2)
      if self._use_state :
        user = tf.squeeze(states[:, 0, :, :], axis=2)
        inputs = tf.concat([user, doc], axis=1)
        q_value_list.append(net(inputs))
      else :
        q_value_list.append(net(doc))

    q_values = tf.concat(q_value_list, axis=1)

    return dqn_agent.DQNNetworkType(q_values)

  def _create_network(self):
    #Create the two online and target networks
    if self._use_state :
      shape=2*self.state.shape[2]
    else :
      shape=self.state.shape[2]
    net=tf.keras.Sequential()
    net.add(tf.keras.Input(shape=shape))
    net.add(tf.keras.layers.Dense(self._network_size, activation=tf.nn.relu))
    net.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
    net.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
    return net

  def _build_networks(self):
    with tf.variable_scope("Online") :  
      self.online_convnet=self._create_network()
      #online replay
      self._replay_net_outputs = self._network_adapter(
            self._replay.states,self.online_convnet)
      #online
      self._net_outputs = self._network_adapter(
            self.state_ph, self.online_convnet)
      #online evaluation
      T=self.state.shape[2]
      if self._use_state :
        diag=tf.cast(tf.linalg.diag(np.ones(T)),tf.float32)
        lignes=[tf.reshape(tf.concat([self._user_ph,diag[i]],axis=0),(1,2*T)) for i in range(T)]
        input=tf.concat(lignes,axis=0)
      else :
        input=tf.cast(tf.linalg.diag(np.ones(T)),tf.float32)
      self._net_q_values = self.online_convnet(input)

    with tf.variable_scope("Target") :  
      self.target_convnet=self._create_network()
      #target replay
      self._replay_next_target_net_outputs = self._network_adapter(
            self._replay.states,self.target_convnet)

    self._build_select_slate_op()

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      An op performing one step of training from replay data.
    """

    # click_indicator: [B, S]
    # q_values: [B, A]
    # actions: [B, S]
    # slate_q_values: [B, S]
    # target : [B]
    # target_slate : [B,S]
    # target_clicked : [B'] 
    # target_clicked_slate : [B',S]
    # slate_q_values: [B,S]
    # slate_q_values_clicked: [B',S]
    
    if not self._cond :
      slate_q_values = tf.batch_gather(
          self._replay_net_outputs.q_values,
          tf.cast(self._replay.actions, dtype=tf.int32))

      target = tf.stop_gradient(self._build_target_q_op())
      if self._loss_calc == 'mse' :
        target_slate = tf.tensordot(tf.reshape(target,(tf.size(target),1)),
                tf.ones((1,self._slate_size)),axes=1)

        loss = tf.reduce_mean(input_tensor=tf.square(slate_q_values - target_slate))

      elif self._loss_calc == 'sum' :
        mean_slate_q_values=tf.reduce_mean(slate_q_values,axis=1)
        loss = tf.reduce_mean(input_tensor=tf.square(mean_slate_q_values - target))

      elif self._loss_calc == 'max' :
        max_slate_q_values=tf.reduce_max(slate_q_values,axis=1)
        loss = tf.reduce_mean(input_tensor=tf.square(max_slate_q_values - target))

    else :
      click_indicator = self._replay.rewards[:, :, self._click_response_index]
      clicked = tf.reduce_sum(input_tensor=click_indicator, axis=1)
      clicked_indices = tf.squeeze(tf.where(tf.equal(clicked, 1)), axis=1)

      target = tf.stop_gradient(self._build_target_q_op())
      target_clicked = tf.gather(target, clicked_indices)



      slate_q_values = tf.batch_gather(
          self._replay_net_outputs.q_values,
          tf.cast(self._replay.actions, dtype=tf.int32))
      slate_q_values_clicked = tf.gather(slate_q_values, clicked_indices)

      def get_train_op() :
        if self._loss_calc == 'mse' :
          target_clicked_slate = tf.tensordot(tf.reshape(target_clicked,(tf.size(target_clicked),1)),
                  tf.ones((1,self._slate_size)),axes=1)
          loss = tf.reduce_mean(input_tensor=tf.square(slate_q_values_clicked - target_clicked_slate))
        elif self._loss_calc == 'sum' :
          mean_slate_q_values_clicked=tf.reduce_mean(slate_q_values_clicked,axis=1)
          loss = tf.reduce_mean(input_tensor=tf.square(mean_slate_q_values_clicked - target_clicked))
        elif self._loss_calc == 'max' :
          max_slate_q_values_clicked=tf.reduce_max(slate_q_values_clicked,axis=1)
          loss = tf.reduce_mean(input_tensor=tf.square(max_slate_q_values_clicked - target_clicked))

        if self.summary_writer is not None :
          with tf.variable_scope('Losses') :
            tf.summary.scalar('Loss', loss)
        return loss

      loss = tf.cond(
          pred=tf.greater(tf.reduce_sum(input_tensor=clicked), 0),
          true_fn=get_train_op,
          false_fn=lambda: tf.constant(0.),
          name='')


    return self.optimizer.minimize(loss)

  def _build_target_q_op(self):
    """Builds an op used as a target for the Q-value.

    Returns:
      An op calculating the Q-value.
    """
    item_reward = self._replay.rewards[:, :, self._reward_response_index]
    click_indicator = self._replay.rewards[:, :, self._click_response_index]
    # Only compute the watch time reward of the clicked item.
    reward = tf.reduce_sum(input_tensor=item_reward * click_indicator, axis=1)

    return self._compute_target_fn(
        reward=reward,
        gamma=self.gamma,
        next_actions=self._replay.next_actions,
        next_q_values=self._replay_next_target_net_outputs.q_values,
        next_states=self._replay.next_states,
        terminals=self._replay.terminals)

  # The following functions defines how the agent takes actions.
  def step(self, reward, observation):
    """Records the transition and returns the agent's next action.

    It uses document-level user response instead of overral reward as the reward
    of the problem.

    Args:
      reward: unused.
      observation: a space.Dict that includes observation of the user state
        observation, documents and user responses.

    Returns:
      Array, the selected action.
    """
    del reward  # Unused argument.

    responses = observation['response']

    if self.user_vector is None :
      self._raw_observation = deepcopy(observation)
    else :
      if (not self.eval_mode and self._learn_state) or self._warm_starting:
        self.user_vector.train_from_row(self._last_action,self._raw_observation['doc'],responses)
      self._raw_observation = deepcopy(observation)
      self._raw_observation['user']=self.user_vector.get_vecteur()[0]
    self._last_action = super(SlateFreeAgent,
                                self).step(self._response_adapter.encode(responses),
                                self._obs_adapter.encode(self._raw_observation))

    return self._last_action

  def _build_select_slate_op(self):
    q = self._net_outputs.q_values[0]
    with tf.name_scope('select_slate'):
      self._output_slate = self._select_slate_fn(self._slate_size, q)

    self._output_slate = tf.reshape(self._output_slate, (self._slate_size,))

    self._action_counts = tf.get_variable(
        'action_counts',
        shape=[self._num_candidates],
        initializer=tf.zeros_initializer())
    output_slate = tf.reshape(self._output_slate, [-1])
    output_one_hot = tf.one_hot(output_slate, self._num_candidates)
    update_ops = []
    for i in range(self._slate_size):
      update_ops.append(tf.assign_add(self._action_counts, output_one_hot[i]))
    self._select_action_update_op = tf.group(*update_ops)

  def _get_q_values(self) :
    user=list(self.state[0,0,:,0])
    q=self._sess.run(self._net_q_values,{self._user_ph:user})[:,0]
    return [float(v) for v in q]

  def _select_action(self):
    """Selects an slate based on the trained model.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates. It will
    pick the top slate_size documents with highest Q values and return them as a
    slate.

    Returns:
       Array, the selected action.
    """
    if self.eval_mode and not self._warm_starting:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                self.min_replay_history, self.epsilon_train)
      self._add_summary('epsilon', epsilon)

    if np.random.random() <= epsilon:
      # Sample without replacement.
      return np.random.choice(
          self._num_candidates, self._slate_size, replace=False)
    else:
      observation = self._raw_observation
      user_obs = observation['user']
      doc_obs = np.array(list(observation['doc'].values()))
      tf.logging.debug('cp 1: %s, %s', doc_obs, observation)
      # TODO(cwhsu): Use score_documents_tf() and remove score_documents().
      output_slate, _ = self._sess.run(
          [self._output_slate, self._select_action_update_op], {
              self.state_ph: self.state
          })

      return output_slate

  # Other functions.
  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A WrapperReplayBuffer object.
    """
    return dqn_agent.wrapped_replay_buffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype,
        action_shape=self._env_action_space.shape,
        action_dtype=self._env_action_space.dtype,
        reward_shape=self._response_adapter.response_shape,
        reward_dtype=self._response_adapter.response_dtype)

  def _add_summary(self, tag, value):
    if self.summary_writer:
      summary = tf.Summary(
          value=[tf.Summary.Value(tag=tag, simple_value=value)])
      self.summary_writer.add_summary(summary, self.training_steps)

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      An integer array of size _slate_size, the selected slated, each
      element of which is an index in the list of doc_obs.
    """
    #print("begin episode")
    self._raw_observation = observation

    return super(SlateFreeAgent,
                 self).begin_episode(self._obs_adapter.encode(observation))

  def end_episode(self, reward, observation):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
      observation: numpy array, the environment's initial observation.
    """
    del reward  # Unused argument.
    super(SlateFreeAgent, self).end_episode(
        self._response_adapter.encode(observation['response']))

  def save(self,dir) :
    """
    Sauvegare data importante
    """
    self.bundle_and_checkpoint(dir,0)
    if self.user_vector is not None :
      self.user_vector.save(dir)

  def load(self,dir) :
    """
    Charge data
    """

    self.unbundle(dir,0,None)
    if self.user_vector is not None :
      self.user_vector.save(dir)

def create_agent(agent_name, sess, **kwargs):
  """Creates a slate decomposition agent given agent name."""
  if agent_name == 'slate_topk_sarsa':
    return SlateFreeAgent(
        sess,
        select_slate_fn=select_slate_topk,
        compute_target_fn=compute_target_sarsa,
        **kwargs)
  if agent_name == 'slate_topk_topk_q':
    return SlateFreeAgent(
        sess,
        select_slate_fn=select_slate_topk,
        compute_target_fn=compute_target_topk_q,
        **kwargs)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))
