"""
model.py

Base neural network model for PowerTAC agent
Inputs:
    Cash Position [0]
    Time of day of current timeslot cosine [1]
    Day of week of current timeslot cosine [2]
    Time of day of current timeslot sine [3]
    Day of week of current timeslot sine [4]

    Latest cleared trade price for 24 bidding timeslot [5-28]
    Latest cleared trade quantity for 24 bidding timeslot [29-52]
    Quantity of uncleared bids for 24 bidding timeslot [53-76]
    Average uncleared bid price for 24 bidding timeslot [77-100]
    Quantity of uncleared ask for 24 bidding timeslot [101-124]
    Average uncleared ask price for 24 bidding timeslot [125-148]
    Electricity balance for 24 bidding timeslot [149-172]
Outputs:
    Market Outputs:
        24 continuous valued policy representing price value to perform bid/ask
        24 continuous valued policy representing quantity value to perform bid/ask
    1 linear output representing V(s)
"""
import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.initializers as init
import tensorflow.contrib.cudnn_rnn as rnn
import numpy as np


def __build_dense__(inputs, num_units, activation=None, initializer=init.orthogonal(),
                    bias_initializer=init.zeros(), name="dense"):
    return layers.dense(inputs, num_units, activation=activation, kernel_initializer=initializer,
                        bias_initializer=bias_initializer, name=name)


class Policy:

    def __init__(self, name):
        self.Name = name

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class CategoricalPolicy(Policy):

    def __init__(self, inputs, num_categories, name="CategoricalPolicy"):
        super().__init__(name)
        self.Logits = __build_dense__(inputs, num_categories, name=name)
        self.NumCategories = num_categories

    def mode(self):
        return tf.argmax(self.Logits, axis=-1)

    def neglogp(self, x):
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Logits,
                                                          labels=tf.one_hot(x, self.Logits.get_shape().as_list()[-1]))

    def entropy(self):
        a0 = self.Logits - tf.reduce_max(self.Logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        return tf.argmax(self.Logits - tf.log(-tf.log(tf.random_uniform(tf.shape(self.Logits),
                                                                        dtype=self.Logits.dtype))),
                         axis=-1)


class ContinuousPolicy(Policy):

    def __init__(self, inputs, num_outputs, name="ContinuousPolicy"):
        super().__init__(name)
        self.Mean = __build_dense__(inputs, num_outputs, name=name + "_Mean")
        self.LogStd = __build_dense__(inputs, num_outputs, initializer=init.zeros(),
                                      name=name + "_LogStd")
        self.Std = tf.exp(self.LogStd)

    def mode(self):
        return self.Mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x  - self.Mean) / self.Std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.LogStd, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.LogStd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.Mean + self.Std * tf.random_normal(tf.shape(self.Mean))


class Model:
    NUM_ENABLED_TIMESLOT = 24
    ACTION_COUNT = 48
    FEATURE_COUNT = 172
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

            self.Name = name
            self.Inputs = inputs
            self.StateTupleOut = state_tuple_in
            self.MarketPolicies = ContinuousPolicy(internal_state, 2 * Model.NUM_ENABLED_TIMESLOT, "MarketPolicy")
            self.StateValue = __build_dense__(internal_state, 1, name="StateValue")
            self.EvaluationOp = self.MarketPolicies.sample()

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
