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
"""Classes to represent the interest evolution documents and users."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
import gin.tf
from gym import spaces
import numpy as np
import json
from recsim_mod import document
from recsim_mod import user
from recsim_mod import utils
from recsim_mod.simulator import environment
from recsim_mod.simulator import recsim_gym

FLAGS = flags.FLAGS
NB_FEATURES = 20 #Latent space of representation
NB_CATEGORIES = 12 #Numbers of categories

class Embendding ():
  def __init__ (self,data_dir):
    self._q=np.load(data_dir+'q20.npy')
    with open(data_dir+'categories.json', 'r') as f:
      categories=json.load(f)
    self._categories={int(k) : v for k,v in categories.items()}
    with open(data_dir+'item_map.json', 'r') as f:
      item_map=json.load(f)
    self._item_map={int(k) : v for k,v in item_map.items()}
    with open(data_dir+'mean_categories.json', 'r') as f:
      mean_categories=json.load(f)
    self._mean_categories={int(k) : v for k,v in mean_categories.items()}
    

  def vecteur_of(self,itemid) : #itemid -> latent vector


    try :
      categorie = self._categories[itemid]
    except :
      categorie = None

    try : x = self._q[self._item_map[itemid]]
    except :
      try : x = self._mean_categories[categorie]
      except : x = np.zeros(NB_FEATURES)


    y = np.concatenate([np.zeros(NB_CATEGORIES),x], axis = 0)
    if categorie is not None :
      y[categorie-1]=1
    
    return y

  def get_item_ids(self) : #Return itemids of embendded items
    return list(self._item_map.keys())
  
  def itemk (self,itemid) : 
    return self._item_map[itemid]
  
  def category (self,itemid) :
    try : return self._categories[itemid]
    except : return 0
  
  def q (self,itemk) :
    return self._q[itemk]

class IEvResponse(user.AbstractResponse):
  """Class to represent a user's response to a video.

  Attributes:
    clicked: A boolean indicating whether the video was clicked.
    watch_time: A float for fraction of the video watched.
    liked: A boolean indicating whether the video was liked.
    quality: A float indicating the quality of the video.
    cluster_id: A integer representing the cluster ID of the video.
  """

  # The min quality score.
  MIN_QUALITY_SCORE = -100
  # The max quality score.
  MAX_QUALITY_SCORE = 100

  def __init__(self,
               clicked=False,
               watch_time=0.0,
               liked=False,
               quality=0.0,
               cluster_id=0.0):
    """Creates a new user response for a video.

    Args:
      clicked: A boolean indicating whether the video was clicked
      watch_time: A float for fraction of the video watched
      liked: A boolean indicating whether the video was liked
      quality: A float for document quality
      cluster_id: a integer for the cluster ID of the document.
    """
    self.clicked = clicked
    self.watch_time = watch_time
    self.liked = liked
    self.quality = quality
    self.cluster_id = cluster_id

  def create_observation(self):
    return {
        'click': int(self.clicked),
        'watch_time': np.array(self.watch_time),
        'liked': int(self.liked),
        'quality': np.array(self.quality),
        'cluster_id': int(self.cluster_id)
    }

  @classmethod
  def response_space(cls):
    # `clicked` feature range is [0, 1]
    # `watch_time` feature range is [0, MAX_VIDEO_LENGTH]
    # `liked` feature range is [0, 1]
    # `quality`: the quality of the document and range is specified by
    #    [MIN_QUALITY_SCORE, MAX_QUALITY_SCORE].
    # `cluster_id`: the cluster the document belongs to and its range is
    # .  [0, IEvVideo.NUM_FEATURES].
    return spaces.Dict({
        'click':
            spaces.Discrete(2),
        'watch_time':
            spaces.Box(
                low=0.0,
                high=IEvVideo.MAX_VIDEO_LENGTH,
                shape=tuple(),
                dtype=np.float32),
        'liked':
            spaces.Discrete(2),
        'quality':
            spaces.Box(
                low=cls.MIN_QUALITY_SCORE,
                high=cls.MAX_QUALITY_SCORE,
                shape=tuple(),
                dtype=np.float32),
        'cluster_id':
            spaces.Discrete(IEvVideo.NUM_FEATURES)
    })


class IEvVideo(document.AbstractDocument):
  """Class to represent a interest evolution Video.

  Attributes:
    features: A numpy array that stores video features.
    cluster_id: An integer that represents.
    video_length : A float for video length.
    quality: a float the represents document quality.
  """

  # The maximum length of videos.
  MAX_VIDEO_LENGTH = 100.0

  # The number of features to represent each video.
  NUM_FEATURES = NB_FEATURES + NB_CATEGORIES

  def __init__(self,
               doc_id,
               features,
               cluster_id=None,
               video_length=None,
               quality=None):
    """Generates a random set of features for this interest evolution Video."""

    # Document features (i.e. distribution over topics)
    self.features = features

    # Cluster ID
    self.cluster_id = cluster_id

    # Length of video
    self.video_length = video_length

    # Document quality (i.e. trashiness/nutritiousness)
    self.quality = quality

    # doc_id is an integer representing the unique ID of this document
    super(IEvVideo, self).__init__(doc_id)

  def create_observation(self):
    """Returns observable properties of this document as a float array."""
    return self.features

  @classmethod
  def observation_space(cls):
    return spaces.Box(
        shape=(cls.NUM_FEATURES,), dtype=np.float32, low=-1.0, high=1.0)



class UtilityModelVideoSampler(document.AbstractDocumentSampler):
  """Class that samples videos for utility model experiment."""

  def __init__(self,
               embendding,
               implicit,
               doc_ctor=IEvVideo,
               reward_click=1,
               reward_buy=5,
               gamma=1.5,
               **kwargs):
    """Creates a new utility model video sampler.

    Args:
      doc_ctor: A class/constructor for the type of videos that will be sampled
        by this sampler.
      min_utility: A float for the min utility score.
      max_utility: A float for the max utility score.
      video_length: A float for the video_length in minutes.
      **kwargs: other keyword parameters for the video sampler.
    """
    super(UtilityModelVideoSampler, self).__init__(doc_ctor, **kwargs)

    self._reward_click=reward_click
    self._reward_buy=reward_buy
    self._gamma=gamma
    self._implicit=implicit

    self._embendding=embendding

    self._item_ids=embendding.get_item_ids()
    self._clicks=None #pas sûr que ce soir utile
    self._iter=0 #où on se trouve dans la session
    self._rewards=None #array de (rewards \cup vecteur latent)

    self._doc_count=0
    self.eval_mode=False


  def load_session(self,session) :
    self._iter=0 
    self._clicks=session[0]
    feedbacks={} #itemid->reward
    for click in session[0] :
      try : feedbacks[click]+=self._reward_click
      except : feedbacks[click]=self._reward_click
    if len(session)>1 :
      for buy,quantity in session[1] :
        try : feedbacks[buy]+=self._reward_buy*quantity
        except : feedbacks[buy]=self._reward_buy*quantity

    self._rewards=np.array([[reward]+list(self._embendding.vecteur_of(itemid)) for itemid,reward in feedbacks.items()])

  def regression_reward(self,item) :
    D=item-self._rewards[:,1:]
    D=np.square(D).sum(axis=1)
    k=np.argmin(D)

    return self._rewards[k,0]*np.exp(-self._gamma*D[k])

  def sample_document(self,itemid=None):
    if self._rewards is None :
      #We manage this situation because at initialisation of the environment there is a sampling without user
      doc_features = {}
      doc_features['doc_id'] = self._doc_count
      doc_features['cluster_id']=0
      doc_features['features']=np.zeros(NB_FEATURES + NB_CATEGORIES)
      doc_features['video_length']=0
      doc_features['quality']=0
      self._doc_count+=1
      return self._doc_ctor(**doc_features)

    else :
      if itemid is None :
        itemid=np.random.choice(self._item_ids)

      doc_features = {}
      
      doc_features['doc_id'] = itemid

      # Sample a cluster_id. Assumes there are NUM_FEATURE clusters.
      doc_features['cluster_id'] = self._embendding.category(itemid)

      # Features are a 1-hot encoding of cluster id
      doc_features['features'] = self._embendding.vecteur_of(itemid)
      # Fixed video lengths (in minutes)
      doc_features['video_length'] = 4

      # Quality
      doc_features['quality'] = self.regression_reward(doc_features['features'])

      self._doc_count+=1
      return self._doc_ctor(**doc_features)

  def sample_documents(self,num_candidates) :
    """
    Sample documents
    When we are testing or when implicit=True we want the real item to be in the set
    In practice, python dictionary stack items in the order, that's why we shuffle candidate set
    """
    candidate_set=document.CandidateSet()
    if (self.eval_mode or self._implicit) and self._clicks is not None and self._iter<len(self._clicks):
      candidate_set.add_document(self.sample_document(itemid=self._clicks[self._iter]))
    while candidate_set.size() < num_candidates :
      candidate_set.add_document(self.sample_document())

    if self.eval_mode :
      candidate_set.shuffle()

    return candidate_set
class IEvUserState(user.AbstractUserState):
  """Class to represent interest evolution users."""

  # Number of features in the user state representation.
  NUM_FEATURES = NB_FEATURES + NB_CATEGORIES

  def __init__(self):
    """Initializes a new user."""

    self._clicks=None
    self._iter=0

  def create_observation(self):
    """Return an observation of this user's observable state."""
    return np.zeros(NB_FEATURES + NB_CATEGORIES)


  @classmethod
  def observation_space(cls):
    return spaces.Box(
        shape=(cls.NUM_FEATURES,), dtype=np.float32, low=-1.0, high=1.0)



@gin.configurable
class UtilityModelUserSampler(user.AbstractUserSampler):
  """Class that samples users for utility model experiment."""

  def __init__(self,
               user_ctor=IEvUserState,
               **kwargs):
    """Creates a new user state sampler."""
    logging.debug('Initialized UtilityModelUserSampler')
    super(UtilityModelUserSampler, self).__init__(user_ctor, **kwargs)


  def sample_user(self):
    return self._user_ctor()


class IEvUserModel(user.AbstractUserModel):
  """Class to model an interest evolution user.

  Assumes the user state contains:
    - user_interests
    - time_budget
    - no_click_mass
  """

  def __init__(self,
               slate_size,
               embendding,
               response_model_ctor=IEvResponse,
               user_state_ctor=IEvUserState,
               seed=0):
    """Initializes a new user model.

    Args:
      slate_size: An integer representing the size of the slate
      choice_model_ctor: A contructor function to create user choice model.
      response_model_ctor: A constructor function to create response. The
        function should take a string of doc ID as input and returns a
        IEvResponse object.
      user_state_ctor: A constructor to create user state
      no_click_mass: A float that will be passed to compute probability of no
        click.
      seed: A integer used as the seed of the choice model.
      alpha_x_intercept: A float for the x intercept of the line used to compute
        interests update factor.
      alpha_y_intercept: A float for the y intercept of the line used to compute
        interests update factor.

    Raises:
      Exception: if choice_model_ctor is not specified.
    """
    self._embendding=embendding
    super(IEvUserModel, self).__init__(
        response_model_ctor,
        UtilityModelUserSampler(
            user_ctor=user_state_ctor,seed=seed),
        slate_size)

  
  def load_session(self,session) :
    self._user_state._iter=0 
    clicks=session[0]
    
    self._user_state._clicks=[self._embendding.vecteur_of(click) for click in clicks]

  def is_terminal(self):
    """Returns a boolean indicating if the session is over."""
    return self._user_state._iter==len(self._user_state._clicks)

  def soft_reset(self) :
    pass

  def update_state(self, slate_documents, responses):
    pass


  def simulate_response(self, documents):
    """Simulates the user's response to a slate of documents with choice model.

    Args:
      documents: a list of IEvVideo objects

    Returns:
      responses: a list of IEvResponse objects, one for each document
    """
    # List of empty responses
    responses = [self._response_model_ctor() for _ in documents]

    # Sample some clicked responses using user's choice model and populate
    # responses.

    ## Choisir le argmin d

    doc_obs = np.array([doc.create_observation() for doc in documents])
    if self._user_state._iter==len(self._user_state._clicks) :
      selected_index = None
    else :
      click=self._user_state._clicks[self._user_state._iter]  
      D=click-doc_obs
      D=np.square(D).sum(axis=1)
      selected_index =np.argmin(D)

    for i, response in enumerate(responses):
      response.quality = documents[i].quality
      response.cluster_id = documents[i].cluster_id

    if selected_index is None:
      return responses
    self._generate_click_response(documents[selected_index],
                                  responses[selected_index])

    return responses

  def _generate_click_response(self, doc, response):
    """Generates a response to a clicked document.

    Right now we assume watch_time is a fixed value that is the minium value of
    time_budget and video_length. In the future, we may want to try more
    variations of watch_time definition.

    Args:
      doc: an IEvVideo object
      response: am IEvResponse for the document
    Updates: response, with whether the document was clicked, liked, and how
      much of it was watched
    """

    response.clicked = True
    response.watch_time = doc.video_length

def clicked_watchtime_reward(responses):
  """Calculates the total clicked watchtime from a list of responses.

  Args:
    responses: A list of IEvResponse objects

  Returns:
    reward: A float representing the total watch time from the responses
  """
  reward = 0.0
  for response in responses:
    if response.clicked:
      reward += response.watch_time
  return reward

def clicked_quality_reward(responses):
  """Calculates the total clicked quality from a list of responses.

  Args:
    responses: A list of IEvResponse objects

  Returns:
    reward: A float representing the total quality from the responses
  """
  reward = 0.0
  for response in responses:
    if response.clicked:
      reward += response.quality
  return reward


def create_environment(env_config):
  """Creates an interest evolution environment."""

  embendding=Embendding(env_config['data_dir'])

  user_model = IEvUserModel(
      env_config['slate_size'],
      response_model_ctor=IEvResponse,
      user_state_ctor=IEvUserState,
      seed=env_config['seed'],
      embendding=embendding)

  document_sampler = UtilityModelVideoSampler(
      doc_ctor=IEvVideo, embendding=embendding,seed=env_config['seed'],
      implicit=env_config['implicit'])

  ievenv = environment.Environment(
      user_model,
      document_sampler,
      env_config['num_candidates'],
      env_config['slate_size'],
      resample_documents=True)
  if env_config['implicit'] :
    return recsim_gym.RecSimGymEnv(ievenv, clicked_watchtime_reward,
                                  utils.aggregate_video_cluster_metrics,
                                  utils.write_video_cluster_metrics)
  else :
    return recsim_gym.RecSimGymEnv(ievenv, clicked_quality_reward,
                                  utils.aggregate_video_cluster_metrics,
                                  utils.write_video_cluster_metrics)