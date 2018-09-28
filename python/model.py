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

    Latest cleared trade price for 24 bidding timeslot [8-31]
    Latest cleared trade quantity for 24 bidding timeslot [32-55]
    Quantity of uncleared bids for 24 bidding timeslot [56-79]
    Average uncleared bid price for 24 bidding timeslot [80-103]
    Quantity of uncleared ask for 24 bidding timeslot [104-127]
    Average uncleared ask price for 24 bidding timeslot [128-151]
    Electricity balance for 24 bidding timeslot [152-175]

    Bootstrap customer count per PowerType [176 - 189]
    Bootstrap customer power usage per PowerType [190 - 203]

    Tariff PowerType category [204 - 218]
    Total number of subscribers for tariff [219]
    Average power usage per customer for tariff in the previous timeslot [220]
    Time of use rate [221 - 232]
    Time of use curtailment ratio [233 - 244]
    Tariff fixed rate [245]
    Maximum curtailment value [246]
    Early withdrawal penalty [247]
    Periodic payment [248]
    Minimum duration [249]

    Same structure as above section x 4 [250 - 295] [296 - 341] [342 - 387] [388 - 433]

Outputs:
    Market Outputs:
        24 category valued policy representing None, Buy, Sell
        24 continuous valued policy representing price value to perform bid/ask
        24 continuous valued policy representing quantity value to perform bid/ask

    Tariff Outputs:
        1 category valued policy representing Tariff Number [0 - 5]
        1 category valued policy representing None, Revoke, Balancing Order, Activate
        1 category valued policy representing Tariff PowerType
        12 continuous valued policy representing Time Of Use Tariff for each 2 hour timeslot for a day
        12 continuous valued policy representing curtailment ratio for each 2 hour timeslot for a day
        5 continuous valued policy representing FIXED_RATE, CURTAILMENT_RATIO, PP, EWP, MD

    1 linear output representing V(s)
"""
import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.initializers as init
import tensorflow.contrib.cudnn_rnn as rnn
import tensorflow.distributions as dist
import numpy as np
from functools import reduce


def __build_dense__(inputs, num_units, activation=None, initializer=init.orthogonal(),
                    bias_initializer=init.zeros(), name="dense"):
    return layers.dense(inputs, num_units, activation=activation, kernel_initializer=initializer,
                        bias_initializer=bias_initializer, name=name)


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

        return tf.reduce_sum(tf.concat(neglogp, axis=1), axis=1, keepdims=True)

    def entropy(self):
        return tf.reduce_sum(tf.concat([policy.entropy() for policy in self.PolicyGroup], axis=-1), axis=-1,
                             keepdims=True)

    def sample(self):
        return tf.concat([policy.sample() for policy in self.PolicyGroup], axis=-1)


class CategoricalPolicy(Policy):

    def __init__(self, inputs, num_categories, name="CategoricalPolicy"):
        super().__init__(name)
        self.Logits = __build_dense__(inputs, num_categories, name=name)
        self.NumCategories = num_categories

    def num_outputs(self):
        return 1

    def mode(self):
        return tf.cast(tf.argmax(self.Logits, axis=-1), tf.float32) + 0.25

    def neglogp(self, x):
        nlp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Logits,
                                                         labels=tf.one_hot(tf.cast(tf.squeeze(x, axis=1), tf.int64),
                                                                           self.Logits.get_shape().as_list()[-1]))
        return tf.expand_dims(nlp, axis=1)

    def entropy(self):
        a0 = self.Logits - tf.reduce_max(self.Logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return -tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1, keepdims=True)

    def sample(self):
        rand_sample = self.Logits - tf.log(-tf.log(tf.random_uniform(tf.shape(self.Logits), dtype=self.Logits.dtype)))
        return tf.expand_dims(tf.cast(tf.argmax(rand_sample, axis=-1), tf.float32) + 0.25, axis=1)


class ContinuousPolicy(Policy):

    def __init__(self, inputs, num_outputs, scale=1.0, name="ContinuousPolicy"):
        super().__init__(name)
        self.Alpha = __build_dense__(inputs, num_outputs, activation=tf.nn.softplus, name=name + "_Alpha") + 1
        self.Beta = __build_dense__(inputs, num_outputs, activation=tf.nn.softplus, name=name + "_Beta") + 1
        self.Distribution = dist.Beta(self.Alpha, self.Beta)
        self.Scale = scale

    def num_outputs(self):
        return self.Alpha.shape.dims[-1]

    def mode(self):
        return ((self.Alpha - 1.0) / (self.Alpha + self.Beta - 2.0)) * self.Scale

    def neglogp(self, x):
        return -self.Distribution.log_prob(x / self.Scale)

    def entropy(self):
        return self.Distribution.entropy()

    def sample(self):
        return self.Distribution.sample() * self.Scale


class Model:
    NUM_ENABLED_TIMESLOT = 24
    ACTION_COUNT = 104
    FEATURE_COUNT = 433
    HIDDEN_STATE_COUNT = 1024

    def __init__(self, inputs, state_tuple_in, name="Model"):
        if isinstance(inputs, int):
            inputs = tf.placeholder(shape=(1, inputs, Model.FEATURE_COUNT), dtype=tf.float32, name="InputPlaceholder")
        elif not isinstance(inputs, tf.Tensor) or \
                not (inputs.shape.ndims == 3 and inputs.shape.dims[-1] == Model.FEATURE_COUNT):
            raise ValueError("Invalid parameter value for inputs")

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            lstm_cell = rnn.CudnnLSTM(1, Model.HIDDEN_STATE_COUNT, kernel_initializer=init.orthogonal(), name=name)

            internal_state, state_tuple_out = lstm_cell(inputs, state_tuple_in)
            internal_state = tf.reshape(internal_state, [-1, Model.HIDDEN_STATE_COUNT])

            market_actions = GroupedPolicy([CategoricalPolicy(internal_state, 3, name="MarketAction")
                                            for _ in range(Model.NUM_ENABLED_TIMESLOT)])
            market_prices = ContinuousPolicy(internal_state, Model.NUM_ENABLED_TIMESLOT, scale=50.0, name="MarketPrice")
            market_quantities = ContinuousPolicy(internal_state, Model.NUM_ENABLED_TIMESLOT, scale=5000.0,
                                                 name="MarketQuantity")
            market_policies = GroupedPolicy([market_actions, market_prices, market_quantities], name="MarketPolicy")

            self.Name = name
            self.Inputs = inputs
            self.StateTupleOut = state_tuple_out
            self.Policies = GroupedPolicy([market_policies,
                                           GroupedPolicy([CategoricalPolicy(internal_state, 5, name="TariffNumber"),
                                                          CategoricalPolicy(internal_state, 4, name="TariffAction"),
                                                          CategoricalPolicy(internal_state, 14, name="PowerType"),
                                                          ContinuousPolicy(internal_state, 12, scale=0.3, name="VF"),
                                                          ContinuousPolicy(internal_state, 12, name="VR"),
                                                          ContinuousPolicy(internal_state, 1, scale=0.3, name="FF"),
                                                          ContinuousPolicy(internal_state, 1, name="FR"),
                                                          ContinuousPolicy(internal_state, 1, scale=2.5, name="PP"),
                                                          ContinuousPolicy(internal_state, 1, scale=50.0, name="EWP"),
                                                          ContinuousPolicy(internal_state, 1, scale=168.0, name="MD")],
                                                         name="TariffPolicy")],
                                          name="Policy")
            self.StateValue = __build_dense__(internal_state, 1, name="StateValue")
            self.EvaluationOp = self.Policies.sample()

    @property
    def variables(self):
        return tf.trainable_variables(self.Name)

    @staticmethod
    def create_transfer_op(srcmodel, dstmodel):
        srcvars = tf.trainable_variables(srcmodel.Name)
        dstvars = tf.trainable_variables(dstmodel.Name)

        return tuple(tf.assign(dstvars[idx], srcvars[idx]) for idx in range(len(srcvars)))

    @staticmethod
    def state_shapes(batch_size):
        return [[1, batch_size, Model.HIDDEN_STATE_COUNT]] * 2
