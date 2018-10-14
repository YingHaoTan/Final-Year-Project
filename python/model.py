"""
model.py

Base neural network model for PowerTAC agent
Inputs:
    Cash Position [0]
    Time of day of current timeslot cosine [1]
    Day of week of current timeslot cosine [2]
    Time of day of current timeslot sine [3]
    Day of week of current timeslot sine [4]
    Balancing transaction charge [5]
    Distribution transaction charge [6]
    Capacity transaction charge [7]
    Temperature, Cloud Cover, Wind Speed, Wind Direction x 25 [8 - 107]

    Latest cleared trade price for 24 bidding timeslot [108-131]
    Latest cleared trade quantity for 24 bidding timeslot [132-155]
    Quantity of uncleared bids for 24 bidding timeslot [156-179]
    Average uncleared bid price for 24 bidding timeslot [180-203]
    Quantity of uncleared ask for 24 bidding timeslot [204-227]
    Average uncleared ask price for 24 bidding timeslot [228-251]
    Electricity balance for 24 bidding timeslot [252-275]

    Bootstrap customer count per PowerType [276 - 288]
    Bootstrap customer power usage per PowerType [289 - 301]

    Tariff PowerType category [302 - 315]
    Total number of subscribers for tariff [316]
    Average power usage per customer for tariff in the previous timeslot [317]
    Time of use rate [318 - 329]
    Time of use regulation rate [330 - 341]
    Tariff fixed rate [342]
    Maximum curtailment value [343]
    Up regulation rate [344]
    Down regulation rate [345]
    Up regulation BO [346]
    Down regulation BO [347]
    Early withdrawal penalty [348]
    Periodic payment [349]
    Minimum duration [350]

    Same structure as above section x 19 [351 - 1281]

Outputs:
    Market Outputs:
        24 category valued policy representing None, Buy, Sell
        24 continuous valued policy representing price value to perform bid/ask
        24 continuous valued policy representing quantity value to perform bid/ask

    Tariff Outputs x 5:
        1 category valued policy representing Tariff Number
        1 category valued policy representing None, Revoke, UP_REG_BO, DOWN_REG_BO, Activate
        1 category valued policy representing Tariff PowerType
        12 continuous valued policy representing Time Of Use Tariff for each 2 hour timeslot for a day
        12 continuous valued policy representing curtailment ratio for each 2 hour timeslot for a day
        7 continuous valued policy representing FIXED_RATE, CURTAILMENT_RATIO, UP_REG, DOWN_REG, PP, EWP, MD

    1 linear output representing V(s)
"""
import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.initializers as init
import tensorflow.contrib.layers as clayers
import tensorflow.contrib.cudnn_rnn as rnn
import tensorflow.distributions as dist
from functools import reduce
import numpy as np


def __build_dense__(inputs, num_units, activation=None, initializer=init.orthogonal(),
                    bias_initializer=init.zeros(), regularizer=clayers.l2_regularizer(1e-4),
                    use_layer_norm=False, name="dense"):
    dense = layers.dense(inputs, num_units, activation=None if use_layer_norm else activation,
                         kernel_initializer=initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=regularizer, name=name)

    if use_layer_norm:
        dense = clayers.layer_norm(dense,
                                   scale=activation is not None and activation is not tf.nn.relu,
                                   activation_fn=activation)

    return dense


class Policy:

    def __init__(self, name):
        self.Name = name

    def num_outputs(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class GroupedPolicy(Policy):

    def __init__(self, policy_group, name="GroupedPolicy"):
        super().__init__(name)
        self.PolicyGroup = policy_group

    def num_outputs(self):
        return reduce(lambda x, y: x + y, map(lambda policy: policy.num_outputs(), self.PolicyGroup))

    def mode(self):
        return tf.concat([policy.mode() for policy in self.PolicyGroup], axis=-1)

    def neglogp(self, x):
        neglogp = []
        index = 0
        for policy in self.PolicyGroup:
            nindex = index + policy.num_outputs()
            neglogp.append(policy.neglogp(x[:, index: nindex]))
            index = nindex

        return tf.concat(neglogp, axis=1)

    def entropy(self):
        return tf.reduce_sum(tf.concat([policy.entropy() for policy in self.PolicyGroup], axis=-1), axis=-1,
                             keepdims=True)

    def sample(self):
        return tf.concat([policy.sample() for policy in self.PolicyGroup], axis=-1)


class CategoricalPolicy(Policy):

    def __init__(self, inputs, num_categories, name="CategoricalPolicy"):
        super().__init__(name)
        self.Logits = __build_dense__(inputs, num_categories, regularizer=None, name=name)
        self.NumCategories = num_categories

    def num_outputs(self):
        return 1

    def mode(self):
        return tf.expand_dims(tf.cast(tf.argmax(self.Logits, axis=-1), tf.float32) + 0.25, axis=1)

    def neglogp(self, x):
        nlp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Logits,
                                                         labels=tf.one_hot(tf.cast(tf.squeeze(x, axis=1), tf.int64),
                                                                           self.Logits.get_shape().as_list()[-1]))
        return tf.expand_dims(nlp, axis=1)

    def entropy(self):
        return -tf.reduce_sum(tf.nn.softmax(self.Logits) * tf.nn.log_softmax(self.Logits), axis=-1, keepdims=True)

    def sample(self):
        rand_sample = self.Logits - tf.log(-tf.log(tf.random_uniform(tf.shape(self.Logits), dtype=self.Logits.dtype)))
        return tf.expand_dims(tf.cast(tf.argmax(rand_sample, axis=-1), tf.float32) + 0.25, axis=1)


class ContinuousPolicy(Policy):

    def __init__(self, inputs, num_outputs, scale=1.0, name="ContinuousPolicy"):
        super().__init__(name)
        self.Alpha = __build_dense__(inputs, num_outputs,
                                     activation=tf.nn.softplus, regularizer=None, name=name + "_Alpha") + 1
        self.Beta = __build_dense__(inputs, num_outputs,
                                    activation=tf.nn.softplus, regularizer=None, name=name + "_Beta") + 1
        self.Distribution = dist.Beta(self.Alpha, self.Beta)
        self.Scale = scale

    def num_outputs(self):
        return self.Alpha.shape.dims[-1]

    def mode(self):
        return ((self.Alpha - 1.0) / (self.Alpha + self.Beta - 2.0)) * self.Scale

    def neglogp(self, x):
        return self.Distribution.log_prob(x / self.Scale)

    def entropy(self):
        return self.Distribution.entropy()

    def sample(self):
        return self.Distribution.sample() * self.Scale


class Model:
    NUM_ENABLED_TIMESLOT = 24
    ACTION_COUNT = 242
    FEATURE_COUNT = 1281
    HIDDEN_STATE_COUNT = 2048
    MARKET_POLICY_STATE_COUNT = 768
    TARIFF_POLICY_STATE_COUNT = 256
    STATE_EMBEDDING_COUNT = 512
    WEATHER_EMBEDDING_COUNT = 128
    MARKET_EMBEDDING_COUNT = 256
    TARIFF_EMBEDDING_COUNT = 64
    EMBEDDING_COUNT = 1280
    TARIFF_SLOTS_PER_ACTOR = 4
    TARIFF_ACTORS = 5

    def __init__(self, inputs, state_tuple_in, name="Model"):
        if isinstance(inputs, int):
            inputs = tf.placeholder(shape=(1, inputs, Model.FEATURE_COUNT), dtype=tf.float32, name="InputPlaceholder")
        elif not isinstance(inputs, tf.Tensor) or \
                not (inputs.shape.ndims == 3 and inputs.shape.dims[-1] == Model.FEATURE_COUNT):
            raise ValueError("Invalid parameter value for inputs")

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            hidden_cell = rnn.CudnnLSTM(1, Model.HIDDEN_STATE_COUNT, name="%s_HState" % name)

            inputs_shape = inputs.shape

            embedding = tf.reshape(inputs, [-1, Model.FEATURE_COUNT])
            embedding = tf.split(embedding, [7, 100, 168, 26,
                                             *([49] * Model.TARIFF_SLOTS_PER_ACTOR * Model.TARIFF_ACTORS)],
                                 axis=-1)
            weather_embedding = __build_dense__(embedding[1], Model.WEATHER_EMBEDDING_COUNT,
                                                activation=tf.nn.relu, initializer=init.orthogonal(np.sqrt(2)),
                                                name="WeatherEmbedding")
            market_embedding = __build_dense__(embedding[2],
                                               Model.MARKET_EMBEDDING_COUNT,
                                               activation=tf.nn.relu, initializer=init.orthogonal(np.sqrt(2)),
                                               name="MarketEmbedding")
            tariff_embeddings = [__build_dense__(embedding[-(idx + 1)],
                                                 Model.TARIFF_EMBEDDING_COUNT,
                                                 activation=tf.nn.relu, initializer=init.orthogonal(np.sqrt(2)),
                                                 name="TariffEmbedding")
                                 for idx in range(Model.TARIFF_SLOTS_PER_ACTOR * Model.TARIFF_ACTORS)]
            embedding = tf.concat([embedding[0], weather_embedding, market_embedding, embedding[3], *tariff_embeddings],
                                  axis=-1)

            embedding = __build_dense__(embedding,
                                        Model.EMBEDDING_COUNT,
                                        activation=tf.nn.relu, initializer=init.orthogonal(np.sqrt(2)),
                                        name="HEmbedding")
            embedding = __build_dense__(embedding,
                                        Model.EMBEDDING_COUNT,
                                        activation=tf.nn.relu,
                                        initializer=init.orthogonal(np.sqrt(2)),
                                        name="Embedding")
            reshaped_embedding = tf.reshape(embedding, [inputs_shape.dims[0], -1, embedding.shape.dims[-1]])

            hidden_state, hidden_tuple_out = hidden_cell(reshaped_embedding, tuple(state_tuple_in))
            hidden_state = tf.reshape(hidden_state, [-1, Model.HIDDEN_STATE_COUNT])
            num_policy_states = Model.MARKET_POLICY_STATE_COUNT + Model.TARIFF_POLICY_STATE_COUNT * Model.TARIFF_ACTORS
            policy_state = __build_dense__(hidden_state,
                                           num_policy_states,
                                           activation=tf.nn.relu,
                                           initializer=init.orthogonal(np.sqrt(2)),
                                           name="PState")
            policy_state = tf.split(policy_state, [Model.MARKET_POLICY_STATE_COUNT,
                                                   *([Model.TARIFF_POLICY_STATE_COUNT] * Model.TARIFF_ACTORS)], axis=-1)

            market_actions = GroupedPolicy([CategoricalPolicy(policy_state[0], 3, name="MarketAction")
                                            for _ in range(Model.NUM_ENABLED_TIMESLOT)])
            market_prices = ContinuousPolicy(policy_state[0], Model.NUM_ENABLED_TIMESLOT, scale=50.0,
                                             name="MarketPrice")
            market_quantities = ContinuousPolicy(policy_state[0], Model.NUM_ENABLED_TIMESLOT, scale=5000.0,
                                                 name="MarketQuantity")
            market_policies = GroupedPolicy([market_actions, market_prices, market_quantities], name="MarketPolicy")

            tariff_policies = GroupedPolicy([GroupedPolicy([CategoricalPolicy(policy_state[idx],
                                                                              Model.TARIFF_SLOTS_PER_ACTOR,
                                                                              name="TariffNumber_%d" % idx),
                                                            CategoricalPolicy(policy_state[idx], 5,
                                                                              name="TariffAction_%d" % idx),
                                                            CategoricalPolicy(policy_state[idx], 13,
                                                                              name="PowerType_%d" % idx),
                                                            ContinuousPolicy(policy_state[idx], 12, scale=0.25,
                                                                             name="VF_%d" % idx),
                                                            ContinuousPolicy(policy_state[idx], 12,
                                                                             name="VR_%d" % idx),
                                                            ContinuousPolicy(policy_state[idx], 1, scale=0.25,
                                                                             name="FF_%d" % idx),
                                                            ContinuousPolicy(policy_state[idx], 1,
                                                                             name="FR_%d" % idx),
                                                            ContinuousPolicy(policy_state[idx], 2, scale=50.0,
                                                                             name="REG_%d" % idx),
                                                            ContinuousPolicy(policy_state[idx], 1, scale=5.0,
                                                                             name="PP_%d" % idx),
                                                            ContinuousPolicy(policy_state[idx], 1, scale=20.0,
                                                                             name="EWP_%d" % idx),
                                                            ContinuousPolicy(policy_state[idx], 1, scale=168.0,
                                                                             name="MD_%d" % idx)],
                                                           name="TariffPolicy_%d" % idx)
                                             for idx in range(1, Model.TARIFF_ACTORS + 1)])

            self.Name = name
            self.Inputs = inputs
            self.Embedding = embedding
            self.PolicyState = policy_state
            self.StateTupleOut = hidden_tuple_out
            self.Policies = GroupedPolicy([market_policies, tariff_policies], name="Policy")
            self.StateValue = __build_dense__(__build_dense__(hidden_state,
                                                              Model.STATE_EMBEDDING_COUNT,
                                                              activation=tf.nn.relu,
                                                              initializer=init.orthogonal(np.sqrt(2)),
                                                              name="StateEmbedding"),
                                              1,
                                              name="StateValue")
            self.EvaluationOp = self.Policies.sample()
            self.PredictOp = self.Policies.mode()

    @property
    def variables(self):
        return tf.trainable_variables(self.Name)

    @staticmethod
    def state_shapes(batch_size):
        return [[1, batch_size, Model.HIDDEN_STATE_COUNT]] * 2
