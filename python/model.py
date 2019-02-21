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
    Temperature, Cloud Cover, Wind Speed, Wind Direction(Cosine), Wind Direction(Sine) x 25 [8 - 132]

    Latest cleared trade price for 24 bidding timeslot [133-156]
    Latest cleared trade quantity for 24 bidding timeslot [157-180]
    Quantity of uncleared bids for 24 bidding timeslot [181-204]
    Average uncleared bid price for 24 bidding timeslot [205-228]
    Quantity of uncleared ask for 24 bidding timeslot [229-252]
    Average uncleared ask price for 24 bidding timeslot [253-276]
    Electricity balance for 24 bidding timeslot [277-300]

    Bootstrap customer count per PowerType [301 - 313]
    Bootstrap customer power usage per PowerType [314 - 326]

    Tariff PowerType category [327 - 340]
    Total number of subscribers for tariff [341]
    Average power usage per customer for tariff in the previous timeslot [342]
    Time of use rate [343 - 348]
    Time of use maximum curtailment value [349 - 354]
    Tariff fixed rate [355]
    Fixed rate maximum curtailment value [356]
    Up regulation rate [357]
    Down regulation rate [358]
    Up regulation BO [359]
    Down regulation BO [360]
    Periodic payment [361]
    Early withdrawal penalty [362]

    Same structure as above section x 19 [363 - 1046]

Outputs:
    Market Outputs:
        24 boolean valued policy representing NoOp, Transact
        24 continuous valued policy representing price value to perform bid/ask
        24 continuous valued policy representing quantity value to perform bid/ask

    Tariff Outputs x 5:
        1 category valued policy representing Tariff Number
        1 category valued policy representing None, Revoke, UP_REG_BO, DOWN_REG_BO, Activate
        1 category valued policy representing Tariff PowerType
        6 continuous valued policy representing Time Of Use Tariff for each 4 hour timeslot for a day
        6 continuous valued policy representing curtailment ratio for each 4 hour timeslot for a day
        6 continuous valued policy representing FIXED_RATE, CURTAILMENT_RATIO, UP_REG, DOWN_REG, PP, EWP

    1 linear output representing V(s)
"""
import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.initializers as init
import tensorflow.contrib.layers as clayers
import tensorflow.contrib.cudnn_rnn as rnn
from tensorflow_probability import distributions
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
        input_projection = __build_dense__(inputs, num_units, activation=None,
                                           initializer=init.orthogonal(), bias_initializer=bias_initializer,
                                           use_layer_norm=use_layer_norm, name="Projection")
        encoder = __build_dense__(inputs, inputs.shape.dims[-1], activation=activation,
                                  initializer=initializer, bias_initializer=bias_initializer,
                                  use_layer_norm=use_layer_norm, name="Encoder")

        return input_projection + __build_dense__(encoder, num_units, activation=activation, initializer=initializer,
                                                  bias_initializer=bias_initializer, use_layer_norm=use_layer_norm,
                                                  name="Output")


def __build_conv_embedding__(inputs, num_units, ssizes=(2, 2, 2, 3), res_block_size=2, initial_dim=(8, 24),
                             activation=None, initializer=init.orthogonal(),
                             bias_initializer=init.zeros(), name="ConvEmbedding"):
    num_layers = len(ssizes)
    assert(num_layers % res_block_size == 0)
    
    with tf.variable_scope(name):
        conv_embedding = tf.layers.conv1d(inputs, initial_dim[0], int(inputs.shape.dims[-1] - initial_dim[1] + 1), 1,
                                          data_format='channels_first', activation=None,
                                          kernel_initializer=init.orthogonal(),
                                          bias_initializer=bias_initializer,
                                          name="SpatialProjection")
        projection = conv_embedding

        for idx in range(len(ssizes)):
            num_filters = int(num_units / (2 ** (num_layers - idx - 1)))

            projection = tf.layers.conv1d(projection, num_filters, 3, ssizes[idx],
                                          data_format='channels_first', activation=None,
                                          kernel_initializer=init.orthogonal(),
                                          bias_initializer=bias_initializer,
                                          padding="SAME",
                                          name="Projection/%d" % (idx + 1))
            conv_embedding = tf.layers.conv1d(conv_embedding, num_filters, 3, ssizes[idx],
                                              data_format='channels_first', activation=activation,
                                              kernel_initializer=initializer,
                                              bias_initializer=bias_initializer,
                                              padding="SAME",
                                              name="Convolution/%d" % (idx + 1))
            if idx % res_block_size == 1:
                conv_embedding = conv_embedding + projection
                projection = conv_embedding

        return tf.reshape(conv_embedding, (-1, conv_embedding.shape[-2]))


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
            logprobs.append(tf.reshape(policy.log_prob(x[:, index: nindex]), (-1, 1)))
            index = nindex

        return tf.reduce_sum(tf.concat(logprobs, axis=1), axis=1)

    def entropy(self):
        return tf.reduce_sum(tf.concat([tf.reshape(policy.entropy(), (-1, 1))
                                        for policy in self.PolicyGroup], axis=1), axis=1)

    def sample(self):
        return tf.concat([policy.sample() for policy in self.PolicyGroup], axis=-1)


class BooleanPolicy(Policy):

    def __init__(self, inputs, name="BooleanPolicy"):
        super().__init__(name)
        self.Probability = tf.squeeze(__build_dense__(inputs, 1, name=name), axis=-1)
        self.Distribution = distributions.Bernoulli(logits=self.Probability)
        self.__sample__ = tf.expand_dims(tf.cast(self.Distribution.sample(), tf.float32), axis=1)

    def num_outputs(self):
        return 1

    def mode(self):
        return tf.expand_dims(tf.cast(tf.greater(self.Probability, 0.5), tf.float32), axis=1)

    def entropy(self):
        return self.Distribution.entropy()

    def log_prob(self, x):
        return self.Distribution.log_prob(tf.cast(tf.squeeze(x, axis=-1), tf.int32))

    def sample(self):
        return self.__sample__


class CategoricalPolicy(Policy):

    def __init__(self, inputs, num_categories, name="CategoricalPolicy"):
        super().__init__(name)
        self.Logits = __build_dense__(inputs, num_categories, name=name)
        self.Distribution = distributions.Categorical(logits=self.Logits)
        self.NumCategories = num_categories
        self.__sample__ = tf.expand_dims(tf.cast(self.Distribution.sample(), tf.float32), axis=1)

    def num_outputs(self):
        return 1

    def mode(self):
        return tf.expand_dims(tf.cast(tf.argmax(self.Logits, axis=-1), tf.float32), axis=1)

    def entropy(self):
        return self.Distribution.entropy()

    def log_prob(self, x):
        return self.Distribution.log_prob(tf.cast(tf.squeeze(x, axis=-1), tf.int32))

    def sample(self):
        return self.__sample__


class GaussianPolicy(Policy):

    def __init__(self, inputs, num_outputs, name="ContinuousPolicy"):
        super().__init__(name)
        self.NumOutputs = num_outputs
        self.Mean = __build_dense__(inputs, num_outputs, name=name + "%s/mean" % name)
        self.Std = tf.exp(tf.get_variable(name='%s/std' % name, shape=num_outputs, initializer=init.zeros(),
                                          trainable=True))
        self.Distribution = distributions.MultivariateNormalDiag(loc=self.Mean, scale_diag=self.Std)
        self.__sample__ = self.Distribution.sample()

    def num_outputs(self):
        return self.Mean.shape.dims[-1]

    def mode(self):
        return self.Mean

    def entropy(self):
        return self.Distribution.entropy()

    def log_prob(self, x):
        return self.Distribution.log_prob(x)

    def sample(self):
        return self.__sample__


class BetaPolicy(Policy):

    def __init__(self, inputs, num_outputs, name="BetaPolicy"):
        super().__init__(name)
        concentrations = tf.exp(__build_dense__(inputs, num_outputs * 2, name=name + "%s/concentrations" % name)) + 1
        c0, c1 = tf.split(concentrations, 2, axis=1)

        self.NumOutputs = num_outputs
        self.Concentration0 = c0
        self.Concentration1 = c1
        self.Distribution = distributions.Beta(c1, c0)
        self.__sample__ = self.Distribution.sample()

    def num_outputs(self):
        return self.Concentration0.shape.dims[-1]

    def mode(self):
        return (self.Concentration1 - 1) / (self.Concentration0 + self.Concentration1 - 2)

    def entropy(self):
        return tf.reduce_sum(self.Distribution.entropy(), axis=1)

    def log_prob(self, x):
        return tf.reduce_sum(self.Distribution.log_prob(x), axis=1)

    def sample(self):
        return self.__sample__


class RunningStatistics:

    def __init__(self, num_features, epsilon=1e-8):
        self.Epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float64)
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

    def __call__(self, inputs, mask, *args, **kwargs):
        normalized = tf.cast((tf.cast(inputs, tf.float64) - self.Mean) / tf.sqrt(self.Var + self.Epsilon), tf.float32)
        return tf.transpose(tf.where(mask, tf.transpose(normalized), tf.transpose(inputs)))


class Model:
    NUM_ENABLED_TIMESLOT = 24
    ACTION_COUNT = 177
    FEATURE_COUNT = 1046
    HIDDEN_STATE_COUNT = 1024
    WEATHER_EMBEDDING_COUNT = 128
    MARKET_EMBEDDING_COUNT = 128
    TARIFF_TYPE_EMBEDDING_COUNT = 5
    TARIFF_EMBEDDING_COUNT = 32
    EMBEDDING_COUNT = 512
    TARIFF_SLOTS_PER_ACTOR = 4
    TARIFF_ACTORS = 5

    def __init__(self, inputs, state_in, reset_state, name="Model"):
        if isinstance(inputs, int):
            inputs = tf.placeholder(shape=(1, inputs, Model.FEATURE_COUNT), dtype=tf.float32, name="InputPlaceholder")
        elif not isinstance(inputs, tf.Tensor) or \
                not (inputs.shape.ndims == 3 and inputs.shape.dims[-1] == Model.FEATURE_COUNT):
            raise ValueError("Invalid parameter value for inputs")

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            running_stats = RunningStatistics(Model.FEATURE_COUNT)

            total_number_tariffs = Model.TARIFF_SLOTS_PER_ACTOR * Model.TARIFF_ACTORS
            inputs_shape = inputs.shape

            mask = tf.convert_to_tensor([*([False] * 4), *([True] * 3),
                                         *([True, True, True, False, False] * 25),
                                         *([True] * 194), *([*([False] * 14), *([True] * 22)] * 20)])
            normalized_inputs = tf.reshape(inputs, [-1, Model.FEATURE_COUNT])
            normalized_inputs = running_stats(normalized_inputs, mask)
            normalized_inputs = tf.split(normalized_inputs, [7, 120, 5, 168, 26,
                                                             *([36] * total_number_tariffs)], axis=-1)

            embedding = Model.__build_input_embedding__(normalized_inputs)
            embedding = tf.reshape(embedding, [inputs_shape.dims[0], -1, embedding.shape.dims[-1]])
            hidden_cell = rnn.CudnnGRU(1, Model.HIDDEN_STATE_COUNT, kernel_initializer=init.orthogonal(),
                                       bias_initializer=init.zeros(), name="%s/HState" % name)
            initial_state_var = tf.get_variable(name="%s/IState" % name, shape=self.state_shapes(1),
                                                initializer=init.zeros(), trainable=True)
            reset_state = tf.reshape(reset_state, (-1, 1))
            state_in = tf.to_float(reset_state) * initial_state_var + (1 - tf.to_float(reset_state)) * state_in
            state_in = tf.expand_dims(state_in, axis=0)

            hidden_state, state_out = hidden_cell(embedding, tuple([state_in]))
            hidden_state = tf.reshape(hidden_state, [-1, Model.HIDDEN_STATE_COUNT])

            market_policies = Model.__build_market_policies__(hidden_state)
            tariff_policies = Model.__build_tariff_policies__(hidden_state)

            self.Name = name
            self.Inputs = inputs
            self.Embedding = embedding
            self.RunningStats = running_stats
            self.StateOut = tf.squeeze(state_out[0], axis=0)
            self.InitialState = tf.ones_like(state_in, dtype=tf.float32) * initial_state_var
            self.Policies = GroupedPolicy([market_policies, tariff_policies], name="Policy")
            self.StateValue = __build_dense__(hidden_state, 1, name="StateValue")
            self.EvaluationOp = self.Policies.sample()
            self.PredictOp = self.Policies.mode()

    @ property
    def variables(self):
        return tf.trainable_variables(self.Name)

    @staticmethod
    def __build_input_embedding__(normalized_inputs):
        current_weather = tf.transpose(tf.reshape(normalized_inputs[2], (-1, 1, 5)), (0, 2, 1))
        forecast_embedding = tf.transpose(tf.reshape(normalized_inputs[1], (-1, 24, 5)), (0, 2, 1))
        weather_embedding = tf.concat([current_weather, forecast_embedding], axis=2)
        weather_embedding = __build_conv_embedding__(weather_embedding,
                                                     Model.WEATHER_EMBEDDING_COUNT,
                                                     activation=tf.nn.relu,
                                                     initializer=init.orthogonal(np.sqrt(2)),
                                                     bias_initializer=init.zeros(),
                                                     name="WeatherEmbedding")

        market_embedding = tf.reshape(normalized_inputs[3], (-1, 7, 24))
        market_embedding = __build_conv_embedding__(market_embedding,
                                                    Model.MARKET_EMBEDDING_COUNT,
                                                    activation=tf.nn.relu,
                                                    initializer=init.orthogonal(np.sqrt(2)),
                                                    bias_initializer=init.zeros(),
                                                    name="MarketEmbedding")

        tariff_embeddings = []
        for idx in range(Model.TARIFF_SLOTS_PER_ACTOR * Model.TARIFF_ACTORS):
            tariff_embedding = normalized_inputs[-(idx + 1)]
            tariff_embedding = tf.split(tariff_embedding, [14, 22], axis=-1)
            tariff_embedding[0] = __build_dense__(tariff_embedding[0], Model.TARIFF_TYPE_EMBEDDING_COUNT,
                                                  name="TariffTypeEmbedding")
            tariff_embeddings.append(__build_embedding__(tf.concat(tariff_embedding, axis=-1),
                                                         Model.TARIFF_EMBEDDING_COUNT,
                                                         activation=tf.nn.relu,
                                                         initializer=init.orthogonal(np.sqrt(2)),
                                                         name="TariffEmbedding"))

        tariff_embeddings = tf.concat(tariff_embeddings, axis=-1)
        embedding = tf.concat([normalized_inputs[0], normalized_inputs[4],
                               weather_embedding,
                               market_embedding,
                               tariff_embeddings], axis=-1)

        embedding = __build_dense__(embedding,
                                    Model.EMBEDDING_COUNT,
                                    activation=tf.nn.relu,
                                    initializer=init.orthogonal(np.sqrt(2)),
                                    name="Embedding")

        return embedding

    @staticmethod
    def __build_market_policies__(state):
        market_actions = GroupedPolicy([BooleanPolicy(state, name="MarketAction")
                                        for _ in range(Model.NUM_ENABLED_TIMESLOT)])
        market_prices = GaussianPolicy(state, Model.NUM_ENABLED_TIMESLOT, name="MarketPrice")
        market_quantities = GaussianPolicy(state, Model.NUM_ENABLED_TIMESLOT, name="MarketQuantity")
        market_policies = GroupedPolicy([market_actions, market_prices, market_quantities], name="MarketPolicy")

        return market_policies

    @staticmethod
    def __build_tariff_policies__(state):
        tariff_policies = []
        for idx in range(1, Model.TARIFF_ACTORS + 1):
            t_number = CategoricalPolicy(state, Model.TARIFF_SLOTS_PER_ACTOR, name="TariffNumber_%d" % idx)
            t_action = CategoricalPolicy(state, 5, name="TariffAction_%d" % idx)
            p_type = CategoricalPolicy(state, 13, name="PowerType_%d" % idx)

            v_f = GaussianPolicy(state, 6, name="VF_%d" % idx)
            v_c = BetaPolicy(state, 6, name="VC_%d" % idx)
            f_f = GaussianPolicy(state, 1, name="FF_%d" % idx)
            f_c = BetaPolicy(state, 1, name="FC_%d" % idx)
            reg = GaussianPolicy(state, 2, name="REG_%d" % idx)
            pp = GaussianPolicy(state, 1, name="PP_%d" % idx)
            ewp = GaussianPolicy(state, 1, name="EWP_%d" % idx)

            tariff_policies.append(GroupedPolicy([t_number, t_action, p_type, v_f, v_c, f_f, f_c, reg, pp, ewp],
                                                 name="TariffPolicy_%d" % idx))
        tariff_policies = GroupedPolicy(tariff_policies, name="TariffPolicies")

        return tariff_policies

    @staticmethod
    def create_transfer_op(srcmodel, dstmodel):
        srcvars = tf.trainable_variables(srcmodel.Name)
        dstvars = tf.trainable_variables(dstmodel.Name)

        return tuple(tf.assign(dstvars[idx], srcvars[idx]) for idx in range(len(srcvars)))

    @staticmethod
    def state_shapes(batch_size):
        return [batch_size, Model.HIDDEN_STATE_COUNT]
