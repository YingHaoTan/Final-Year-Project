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
    Time of use maximum curtailment value [330 - 341]
    Tariff fixed rate [342]
    Fixed rate maximum curtailment value [343]
    Up regulation rate [344]
    Down regulation rate [345]
    Up regulation BO [346]
    Down regulation BO [347]
    Periodic payment [348]

    Same structure as above section x 19 [349 - 1241]

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
        5 continuous valued policy representing FIXED_RATE, CURTAILMENT_RATIO, UP_REG, DOWN_REG, PP

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
                    bias_initializer=init.zeros(), use_layer_norm=False, name="dense"):
    dense = layers.dense(inputs, num_units, activation=None if use_layer_norm else activation,
                         kernel_initializer=initializer, bias_initializer=bias_initializer,
                         name=name)

    if use_layer_norm:
        dense = clayers.layer_norm(dense,
                                   scale=activation is not None and activation is not tf.nn.relu,
                                   activation_fn=activation)

    return dense


def __build_embedding__(inputs, num_units, activation=None, initializer=init.orthogonal(),
                        bias_initializer=init.zeros(), use_layer_norm=False, name="Embedding"):
    with tf.variable_scope(name):
        encoder = __build_dense__(inputs, inputs.shape[-1], activation=activation,
                                  initializer=initializer, bias_initializer=bias_initializer,
                                  use_layer_norm=use_layer_norm, name="Encoder")

        return __build_dense__(encoder, num_units, activation=activation, initializer=initializer,
                               bias_initializer=bias_initializer, use_layer_norm=use_layer_norm, name="Vector")


class Policy:

    def __init__(self, name):
        self.Name = name

    def num_outputs(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def log_prob(self, x):
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

    def log_prob(self, x):
        logprobs = []
        index = 0
        for policy in self.PolicyGroup:
            nindex = index + policy.num_outputs()
            logprobs.append(policy.log_prob(x[:, index: nindex]))
            index = nindex

        return tf.concat(logprobs, axis=1)

    def entropy(self):
        return tf.reduce_sum(tf.concat([policy.entropy() for policy in self.PolicyGroup], axis=-1), axis=-1,
                             keepdims=True)

    def sample(self):
        return tf.concat([policy.sample() for policy in self.PolicyGroup], axis=-1)


class CategoricalPolicy(Policy):

    def __init__(self, inputs, num_categories, name="CategoricalPolicy"):
        super().__init__(name)
        self.Logits = __build_dense__(inputs, num_categories, name=name)
        self.Distribution = dist.Categorical(logits=self.Logits)
        self.NumCategories = num_categories

    def num_outputs(self):
        return 1

    def mode(self):
        return tf.expand_dims(tf.cast(tf.argmax(self.Logits, axis=-1), tf.float32) + 0.25, axis=1)

    def log_prob(self, x):
        return tf.expand_dims(self.Distribution.log_prob(tf.cast(tf.squeeze(x, axis=-1), tf.int32)), axis=1)

    def entropy(self):
        return tf.expand_dims(self.Distribution.entropy(), axis=1)

    def sample(self):
        return tf.expand_dims(tf.cast(self.Distribution.sample(), tf.float32) + 0.25, axis=1)


class ContinuousPolicy(Policy):

    def __init__(self, inputs, num_outputs, scale=1.0, name="ContinuousPolicy"):
        super().__init__(name)
        self.Alpha = __build_dense__(inputs, num_outputs,
                                     activation=tf.nn.softplus,
                                     name=name + "_Alpha") + 1
        self.Beta = __build_dense__(inputs, num_outputs,
                                    activation=tf.nn.softplus,
                                    name=name + "_Beta") + 1
        self.Distribution = dist.Beta(self.Alpha, self.Beta)
        self.Scale = scale

    def num_outputs(self):
        return self.Alpha.shape.dims[-1]

    def mode(self):
        return ((self.Alpha - 1.0) / (self.Alpha + self.Beta - 2.0)) * self.Scale

    def log_prob(self, x):
        return self.Distribution.log_prob(x / self.Scale)

    def entropy(self):
        return self.Distribution.entropy() + tf.log(self.Scale)

    def sample(self):
        return self.Distribution.sample() * self.Scale


class RunningStatistics:

    def __init__(self, num_features, epsilon=1e-8):
        self.Epsilon = epsilon
        self.MeanInput = tf.placeholder(name="mean", shape=(1, num_features), dtype=tf.float64)
        self.VarInput = tf.placeholder(name="variance", shape=(1, num_features), dtype=tf.float64)
        self.CountInput = tf.placeholder(name="count", shape=(), dtype=tf.int64)

        with tf.variable_scope("RunningStatistics", reuse=tf.AUTO_REUSE):
            self.Mean = tf.get_variable("mean", shape=(1, num_features), initializer=init.zeros(),
                                        trainable=False, dtype=tf.float64)
            self.Var = tf.get_variable("variance", shape=(1, num_features), initializer=init.ones(),
                                       trainable=False, dtype=tf.float64)
            self.Count = tf.get_variable("count", shape=(), initializer=init.zeros(),
                                         trainable=False, dtype=tf.int64)

        self.UpdateStatsOp = tf.cond(self.Count > 0, lambda: self.__update_stats__(), lambda: self.__reset_stats__())

    def __reset_stats__(self):
        with tf.control_dependencies([tf.assign(self.Mean, self.MeanInput), tf.assign(self.Var, self.VarInput)]):
            return tf.assign(self.Count, self.CountInput)

    def __update_stats__(self):
        count = self.CountInput + self.Count
        count_input_f, count_f, new_count_f = (tf.cast(self.CountInput, tf.float64), tf.cast(self.Count, tf.float64),
                                               tf.cast(count, tf.float64))

        delta = self.MeanInput - self.Mean
        mean = self.Mean + delta * count_input_f / new_count_f
        m_a = self.Var * count_f
        m_b = self.VarInput * count_input_f
        m = m_a + m_b + tf.square(delta) * count_f * count_input_f / new_count_f
        var = m / new_count_f

        with tf.control_dependencies([tf.assign(self.Mean, mean), tf.assign(self.Var, var)]):
            return tf.assign(self.Count, count)

    def __call__(self, inputs, *args, **kwargs):
        return tf.cast((tf.cast(inputs, tf.float64) - self.Mean) / tf.sqrt(self.Var + self.Epsilon), tf.float32)


class Model:
    NUM_ENABLED_TIMESLOT = 24
    ACTION_COUNT = 232
    FEATURE_COUNT = 1241
    HIDDEN_STATE_COUNT = 2048
    MARKET_COV_STATE_COUNT = 512
    TARIFF_COV_STATE_COUNT = 128
    WEATHER_EMBEDDING_COUNT = 128
    MARKET_EMBEDDING_COUNT = 256
    TARIFF_EMBEDDING_COUNT = 64
    EMBEDDING_COUNT = 1024
    TARIFF_SLOTS_PER_ACTOR = 4
    TARIFF_ACTORS = 5

    def __init__(self, inputs, state_tuple_in, name="Model"):
        if isinstance(inputs, int):
            inputs = tf.placeholder(shape=(1, inputs, Model.FEATURE_COUNT), dtype=tf.float32, name="InputPlaceholder")
        elif not isinstance(inputs, tf.Tensor) or \
                not (inputs.shape.ndims == 3 and inputs.shape.dims[-1] == Model.FEATURE_COUNT):
            raise ValueError("Invalid parameter value for inputs")

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            running_stats = RunningStatistics(Model.FEATURE_COUNT)
            hidden_cell = rnn.CudnnLSTM(1, Model.HIDDEN_STATE_COUNT, name="%s_HState" % name)

            inputs_shape = inputs.shape

            embedding = tf.reshape(inputs, [-1, Model.FEATURE_COUNT])
            embedding = running_stats(embedding)
            embedding = tf.split(embedding, [7, 100, 168, 26,
                                             *([47] * Model.TARIFF_SLOTS_PER_ACTOR * Model.TARIFF_ACTORS)],
                                 axis=-1)
            weather_embedding = __build_embedding__(tf.concat([embedding[0], embedding[1]], axis=-1),
                                                    Model.WEATHER_EMBEDDING_COUNT,
                                                    activation=tf.nn.relu, initializer=init.orthogonal(np.sqrt(2)),
                                                    name="WeatherEmbedding")
            market_embedding = __build_embedding__(tf.concat([embedding[0], embedding[2]], axis=-1),
                                                   Model.MARKET_EMBEDDING_COUNT,
                                                   activation=tf.nn.relu, initializer=init.orthogonal(np.sqrt(2)),
                                                   name="MarketEmbedding")
            tariff_embeddings = [__build_embedding__(tf.concat([embedding[0], embedding[3], embedding[-(idx + 1)]],
                                                               axis=-1),
                                                     Model.TARIFF_EMBEDDING_COUNT,
                                                     activation=tf.nn.relu, initializer=init.orthogonal(np.sqrt(2)),
                                                     name="TariffEmbedding")
                                 for idx in range(Model.TARIFF_SLOTS_PER_ACTOR * Model.TARIFF_ACTORS)]
            tariff_embeddings = tf.concat(tariff_embeddings, axis=-1)
            embedding = tf.concat([embedding[0], weather_embedding, market_embedding, tariff_embeddings], axis=-1)

            embedding = __build_embedding__(embedding,
                                            Model.EMBEDDING_COUNT,
                                            activation=tf.nn.relu,
                                            initializer=init.orthogonal(np.sqrt(2)),
                                            name="Embedding")
            reshaped_embedding = tf.reshape(embedding, [inputs_shape.dims[0], -1, embedding.shape.dims[-1]])

            hidden_state, hidden_tuple_out = hidden_cell(reshaped_embedding, tuple(state_tuple_in))
            hidden_state = tf.reshape(hidden_state, [-1, Model.HIDDEN_STATE_COUNT])
            cov_state_splits = [Model.MARKET_COV_STATE_COUNT,
                                *([Model.TARIFF_COV_STATE_COUNT] * Model.TARIFF_ACTORS * Model.TARIFF_SLOTS_PER_ACTOR)]
            cov_state = __build_dense__(hidden_state,
                                        reduce(lambda a, b: a + b, cov_state_splits),
                                        activation=tf.nn.relu,
                                        initializer=init.orthogonal(np.sqrt(2)),
                                        name="CovarianceState")
            cov_state = tf.split(cov_state, cov_state_splits, axis=-1)

            market_actions = GroupedPolicy([CategoricalPolicy(hidden_state, 3, name="MarketAction")
                                            for _ in range(Model.NUM_ENABLED_TIMESLOT)])
            market_prices = ContinuousPolicy(cov_state[0], Model.NUM_ENABLED_TIMESLOT, scale=50.0,
                                             name="MarketPrice")
            market_quantities = ContinuousPolicy(cov_state[0], Model.NUM_ENABLED_TIMESLOT, scale=100.0,
                                                 name="MarketQuantity")
            market_policies = GroupedPolicy([market_actions, market_prices, market_quantities], name="MarketPolicy")

            tariff_policies = GroupedPolicy([GroupedPolicy([CategoricalPolicy(hidden_state,
                                                                              Model.TARIFF_SLOTS_PER_ACTOR,
                                                                              name="TariffNumber_%d" % idx),
                                                            CategoricalPolicy(hidden_state, 5,
                                                                              name="TariffAction_%d" % idx),
                                                            CategoricalPolicy(hidden_state, 13,
                                                                              name="PowerType_%d" % idx),
                                                            ContinuousPolicy(cov_state[idx], 12, scale=0.25,
                                                                             name="VF_%d" % idx),
                                                            ContinuousPolicy(cov_state[idx], 12,
                                                                             name="VR_%d" % idx),
                                                            ContinuousPolicy(cov_state[idx], 1, scale=0.25,
                                                                             name="FF_%d" % idx),
                                                            ContinuousPolicy(cov_state[idx], 1,
                                                                             name="FR_%d" % idx),
                                                            ContinuousPolicy(cov_state[idx], 2, scale=50.0,
                                                                             name="REG_%d" % idx),
                                                            ContinuousPolicy(cov_state[idx], 1, scale=5.0,
                                                                             name="PP_%d" % idx)],
                                                           name="TariffPolicy_%d" % idx)
                                             for idx in range(1, Model.TARIFF_ACTORS + 1)])

            self.Name = name
            self.Inputs = inputs
            self.Embedding = embedding
            self.RunningStats = running_stats
            self.HiddenState = hidden_state
            self.StateTupleOut = hidden_tuple_out
            self.Policies = GroupedPolicy([market_policies, tariff_policies], name="Policy")
            self.StateValue = __build_dense__(hidden_state, 1, name="StateValue")
            self.EvaluationOp = self.Policies.sample()
            self.PredictOp = self.Policies.mode()

    @property
    def variables(self):
        return tf.trainable_variables(self.Name)

    @staticmethod
    def state_shapes(batch_size):
        return [[1, batch_size, Model.HIDDEN_STATE_COUNT]] * 2
