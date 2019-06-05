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

    Tariff PowerType category [327]
    Total number of subscribers for tariff [328]
    Average power usage per customer for tariff in the previous timeslot [329]
    Time of use rate [330 - 335]
    Time of use maximum curtailment value [336 - 341]
    Tariff fixed rate [342]
    Fixed rate maximum curtailment value [343]
    Up regulation rate [344]
    Down regulation rate [345]
    Up regulation BO [346]
    Down regulation BO [347]
    Periodic payment [348]
    Early withdrawal penalty [349]

    Same structure as above section x 19 [350 - 786]

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
from tensorflow import initializers as init
from tensorflow_probability import distributions
from functools import reduce
from typing import Union
import numpy as np


class NetworkModule:
    """Abstract base class for neural network modules"""

    def __init__(self, name=None):
        name = '' if name is None else "%s/" % name if name[-1] != '/' else name

        self.__built__ = False
        self.__scope__ = "%s%s/" % (name, type(self).__name__)
        self.__scope__ = tf.name_scope(self.__scope__)
        self.__parameter_scope__ = tf.name_scope("%s%s/" % (self.__scope__.name, 'parameters'))
        self.__operation_scope__ = tf.name_scope("%s%s/" % (self.__scope__.name, 'operations'))
        self.__weight_map__ = {}

    @property
    def weights(self):
        """Returns a list of trainable weights used by this neural network module"""
        return list(self.__weight_map__.values())

    @property
    def trainable_weights(self):
        return list(filter(lambda x: x.trainable, self.weights))

    def build(self, input_shape: Union[list, tuple]):
        """Build neural network module that with appropriate size to accomodate the specified input shape"""
        self.__built__ = True

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        """Process the input tensor through this neural network module"""
        if not self.__built__:
            self.build(inputs.shape.as_list())

    def __create_weights__(self, initial_value, name):
        with self.__parameter_scope__:
            self.__weight_map__[name] = tf.Variable(initial_value, name=name)


class InputNormalizer(NetworkModule):
    """
        InputNormalizer is a module that aggregates mean and variance statistics
        to perform normalization on the input data
    """

    def __init__(self, mask=None, epsilon=1e-8, name=None):
        super().__init__(name)
        self.__mask__ = mask
        self.__epsilon__ = epsilon
        self.__mean__ = tf.placeholder(tf.float32)
        self.__var__ = tf.placeholder(tf.float32)
        self.__count__ = tf.placeholder(tf.float32)
        self.__update_op__ = None

    @property
    def num_inputs(self):
        if self.__mask__ is not None:
            return self.__mask__.shape.as_list()[0]
        else:
            return None

    def update(self, session, mean, var, count):
        session.run(self.__update_op__, feed_dict={self.__mean__: mean, self.__var__: var, self.__count__: count})

    def build(self, input_shape: list):
        super().build(input_shape)

        with self.__parameter_scope__:
            self.__weight_map__['mean'] = tf.Variable(tf.zeros((1, input_shape[-1])), trainable=False)
            self.__weight_map__['var'] = tf.Variable(tf.zeros((1, input_shape[-1])), trainable=False)
            self.__weight_map__['count'] = tf.Variable(tf.zeros(()), trainable=False)

            mean, var, count = tf.cond(self.__weight_map__['count'] > 0.0,
                                       lambda: self.__calculate_statistics__(),
                                       lambda: (self.__mean__, self.__var__, self.__count__))
            self.__update_op__ = tf.group([
                tf.assign(self.__weight_map__['mean'], mean),
                tf.assign(self.__weight_map__['var'], var),
                tf.assign(self.__weight_map__['count'], count)
            ])

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        super().__call__(inputs, *args, **kwargs)

        with self.__operation_scope__:
            outputs = (inputs - self.__weight_map__['mean']) / tf.sqrt(self.__weight_map__['var'] + self.__epsilon__)
            if self.__mask__ is not None:
                broadcast_mask = tf.tile(tf.expand_dims(self.__mask__, axis=0),
                                         tf.concat([tf.shape(inputs)[:1], tf.ones(1, tf.int32)], axis=0))
                outputs = tf.where(broadcast_mask, outputs, inputs)

        return outputs

    def __calculate_statistics__(self):
        count = self.__count__ + self.__weight_map__['count']
        count_input_f, count_f, new_count_f = (self.__count__, self.__weight_map__['count'], count)

        delta = self.__mean__ - self.__weight_map__['mean']
        mean = self.__weight_map__['mean'] + delta * count_input_f / new_count_f
        m_a = self.__weight_map__['var'] * count_f
        m_b = self.__var__ * count_input_f
        m = m_a + m_b + tf.square(delta) * count_f * count_input_f / new_count_f
        var = m / new_count_f

        return mean, var, count


class ConvolutionEncoder(NetworkModule):
    """ConvolutionEncoder to encode inputs using convolutional neural network"""

    def __init__(self, num_output, initial_dim=(8, 24),
                 ssizes=(2, 2, 2, 3), res_block_size=2, activation=tf.nn.relu,
                 kernel_initializer=init.orthogonal(np.sqrt(2)), bias_initializer=init.zeros(),
                 name=None):
        super().__init__(name)
        self.__num_output__ = num_output
        self.__num_layers__ = len(ssizes)
        self.__projection_size__ = initial_dim
        self.__strides__ = ssizes
        self.__block_size__ = res_block_size
        self.__activation__ = activation
        self.__kernel_initializer__ = kernel_initializer
        self.__bias_initializer__ = bias_initializer

    def build(self, input_shape: Union[list, tuple]):
        """
            Input shape format in NCH
            ConvolutionEncoder kernel upgrades the shape to NCH1 to perform convolution
        """
        super().build(input_shape)
        l_initializer = init.orthogonal()
        k_initializer = self.__kernel_initializer__
        b_initializer = self.__bias_initializer__

        kernel_size = (int(input_shape[-1] - self.__projection_size__[1] + 1), 1,
                       input_shape[1], self.__projection_size__[0])
        self.__create_weights__(l_initializer(kernel_size), 'initial_projection_kernel')
        self.__create_weights__(b_initializer(self.__projection_size__[0]), 'initial_projection_bias')

        projection_in_c = self.__projection_size__[0]
        projection_out_k = 1
        for idx in range(self.__num_layers__):
            out_c = int(self.__num_output__ / (2 ** (self.__num_layers__ - idx - 1)))

            kernel_size = (3, 1, kernel_size[-1], out_c)
            self.__create_weights__(k_initializer(kernel_size), "conv_kernel_%d" % idx)
            self.__create_weights__(b_initializer(out_c), "conv_bias_%d" % idx)

            projection_out_k = projection_out_k * self.__strides__[idx]

            if idx % self.__block_size__ == 1:
                residx = idx // self.__block_size__

                p_kernel_size = (projection_out_k, 1, projection_in_c, out_c)
                self.__create_weights__(l_initializer(p_kernel_size), "projection_kernel_%d" % residx)
                self.__create_weights__(b_initializer(out_c), "projection_bias_%d" % residx)

                initial_gate_value = tf.ones((1, out_c, 1, 1))
                self.__create_weights__(initial_gate_value, "cgate_kernel_%d" % residx)

                projection_in_c = out_c
                projection_out_k = 1

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        super().__call__(inputs, *args, **kwargs)

        with self.__operation_scope__:
            expanded_inputs = tf.expand_dims(inputs, axis=-1)

            conv_output = tf.nn.bias_add(tf.nn.conv2d(expanded_inputs, self.__weight_map__['initial_projection_kernel'],
                                                      [1, 1, 1, 1], padding='VALID', data_format='NCHW',
                                                      name='SpatialProjection'),
                                         self.__weight_map__['initial_projection_bias'],
                                         data_format='NCHW')

            projection = conv_output
            projection_stride = 1
            for idx in range(self.__num_layers__):
                conv_output = tf.nn.bias_add(tf.nn.conv2d(conv_output,
                                                          self.__weight_map__["conv_kernel_%d" % idx],
                                                          [1, 1, self.__strides__[idx], 1],
                                                          padding='SAME', data_format='NCHW',
                                                          name="Conv%d" % idx),
                                             self.__weight_map__["conv_bias_%d" % idx],
                                             data_format='NCHW')
                conv_output = self.__activation__(conv_output)
                projection_stride = projection_stride * self.__strides__[idx]

                if idx % self.__block_size__ == 1:
                    residx = idx // self.__block_size__

                    projection = tf.nn.bias_add(tf.nn.conv2d(projection,
                                                             self.__weight_map__["projection_kernel_%d" % residx],
                                                             [1, 1, projection_stride, 1],
                                                             padding='SAME', data_format='NCHW',
                                                             name="Conv%d" % residx),
                                                self.__weight_map__["projection_bias_%d" % residx],
                                                data_format='NCHW')

                    cgate = self.__weight_map__["cgate_kernel_%d" % residx]
                    conv_output = projection = (cgate * conv_output) + projection

            conv_output_shape = conv_output.shape.as_list()
            output = tf.reshape(conv_output, (-1, conv_output_shape[1] * conv_output_shape[2] * conv_output_shape[3]))

        return output


class DepthEncoder(NetworkModule):
    """DepthEncoder to encode the depth/channel of a convolutional neural network"""

    def __init__(self, num_output, num_layers=2, res_block_size=2, activation=tf.nn.relu,
                 kernel_initializer=init.orthogonal(np.sqrt(2)), bias_initializer=init.zeros(),
                 name=None):
        super().__init__(name)
        self.__num_output__ = num_output
        self.__num_layers__ = num_layers
        self.__block_size__ = res_block_size
        self.__activation__ = activation
        self.__kernel_initializer__ = kernel_initializer
        self.__bias_initializer__ = bias_initializer

    def build(self, input_shape: Union[list, tuple]):
        """
            Input shape format in NCH
            DepthEncoder kernel upgrades the shape to NCH1 to perform convolution
        """
        super().build(input_shape)
        l_initializer = init.orthogonal()
        k_initializer = self.__kernel_initializer__
        b_initializer = self.__bias_initializer__

        inc_layer_count = (self.__num_output__ - input_shape[1]) // self.__num_layers__
        kernel_size = (1, 1, 1, input_shape[1])
        projection_in_c = input_shape[1]
        for idx in range(self.__num_layers__):
            if idx < self.__num_layers__ - 1:
                out_c = kernel_size[-1] + inc_layer_count
            else:
                out_c = self.__num_output__

            kernel_size = (1, 1, kernel_size[-1], out_c)
            self.__create_weights__(k_initializer(kernel_size), "conv_kernel_%d" % idx)
            self.__create_weights__(b_initializer(out_c), "conv_bias_%d" % idx)

            if idx % self.__block_size__ == 1:
                residx = idx // self.__block_size__

                p_kernel_size = (1, 1, projection_in_c, out_c)
                self.__create_weights__(l_initializer(p_kernel_size), "projection_kernel_%d" % residx)
                self.__create_weights__(b_initializer(out_c), "projection_bias_%d" % residx)

                initial_gate_value = tf.ones((1, out_c, 1, 1))
                self.__create_weights__(initial_gate_value, "cgate_kernel_%d" % residx)

                projection_in_c = out_c

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        super().__call__(inputs, *args, **kwargs)

        with self.__operation_scope__:
            expanded_inputs = tf.expand_dims(inputs, axis=-1)

            conv_output = projection = expanded_inputs
            for idx in range(self.__num_layers__):
                conv_output = tf.nn.bias_add(tf.nn.conv2d(conv_output,
                                                          self.__weight_map__["conv_kernel_%d" % idx],
                                                          [1, 1, 1, 1], padding='VALID',
                                                          data_format='NCHW', name="Conv%d" % idx),
                                             self.__weight_map__["conv_bias_%d" % idx],
                                             data_format='NCHW')
                conv_output = self.__activation__(conv_output)

                if idx % self.__block_size__ == 1:
                    residx = idx // self.__block_size__

                    projection = tf.nn.bias_add(tf.nn.conv2d(projection,
                                                             self.__weight_map__["projection_kernel_%d" % residx],
                                                             [1, 1, 1, 1],
                                                             padding='SAME', data_format='NCHW',
                                                             name="Conv%d" % residx),
                                                self.__weight_map__["projection_bias_%d" % residx],
                                                data_format='NCHW')

                    cgate = self.__weight_map__["cgate_kernel_%d" % residx]
                    conv_output = projection = (cgate * conv_output) + projection

            conv_output_shape = conv_output.shape.as_list()
            output = tf.reshape(conv_output, (-1, conv_output_shape[1] * conv_output_shape[2] * conv_output_shape[3]))

        return output


class DenseEncoder(NetworkModule):
    """DenseEncoder to represent a fully connected neural network"""

    def __init__(self, num_output, num_layers=2, res_block_size=2, activation=tf.nn.relu,
                 kernel_initializer=init.orthogonal(np.sqrt(2)), bias_initializer=init.zeros(),
                 name=None):
        super().__init__(name)
        self.__num_output__ = num_output
        self.__num_layers__ = num_layers
        self.__block_size__ = res_block_size
        self.__activation__ = activation
        self.__kernel_initializer__ = kernel_initializer
        self.__bias_initializer__ = bias_initializer

    def build(self, input_shape: list):
        super().build(input_shape)

        l_initializer = init.orthogonal()
        k_initializer = self.__kernel_initializer__
        b_initializer = self.__bias_initializer__

        inc_layer_count = (self.__num_output__ - input_shape[1]) // self.__num_layers__
        kernel_size = (1, input_shape[1])
        projection_in_c = input_shape[1]
        for idx in range(self.__num_layers__):
            if idx < self.__num_layers__ - 1:
                out_c = kernel_size[-1] + inc_layer_count
            else:
                out_c = self.__num_output__

            kernel_size = (kernel_size[-1], out_c)
            self.__create_weights__(k_initializer(kernel_size), "dense_kernel_%d" % idx)
            self.__create_weights__(b_initializer(out_c), "dense_bias_%d" % idx)

            if idx % self.__block_size__ == 1:
                residx = idx // self.__block_size__

                p_kernel_size = (projection_in_c, out_c)
                self.__create_weights__(l_initializer(p_kernel_size), "projection_kernel_%d" % residx)
                self.__create_weights__(b_initializer(out_c), "projection_bias_%d" % residx)

                initial_gate_value = tf.ones((1, out_c))
                self.__create_weights__(initial_gate_value, "cgate_kernel_%d" % residx)

                projection_in_c = out_c

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        super().__call__(inputs)

        with self.__operation_scope__:
            output = projection = inputs
            for idx in range(self.__num_layers__):
                output = self.__activation__(tf.nn.bias_add(tf.matmul(output,
                                                                      self.__weight_map__["dense_kernel_%d" % idx]),
                                                            self.__weight_map__["dense_bias_%d" % idx]))

                if idx % self.__block_size__ == 1:
                    residx = idx // self.__block_size__

                    projection = tf.nn.bias_add(tf.matmul(projection,
                                                          self.__weight_map__["projection_kernel_%d" % residx]),
                                                self.__weight_map__["projection_bias_%d" % residx])

                    cgate = self.__weight_map__["cgate_kernel_%d" % residx]
                    output = projection = (cgate * output) + projection

        return output


class PolicyOutput:

    def log_prob(self, x: tf.Tensor):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError


class DistributionOutput(PolicyOutput):

    def __init__(self, distribution, stop_entropy_gradient=False):
        self.__distribution__ = distribution
        self.__stop_entropy_gradient__ = stop_entropy_gradient

    @staticmethod
    def __conform_shape_ndims__(tensor: tf.Tensor, ndims=1):
        if tensor.shape.ndims > ndims:
            return tf.reduce_sum(tensor, axis=list(range(ndims, tensor.shape.ndims)))
        elif tensor.shape.ndims < ndims:
            return tf.reshape(tensor, list(map(lambda x: -1 if x is None else x, tensor.shape.as_list())) +
                              [1 for _ in range(ndims - tensor.shape.ndims)])
        else:
            return tensor

    def log_prob(self, x: tf.Tensor):
        expected_ndims = self.__distribution__.batch_shape.ndims + self.__distribution__.event_shape.ndims
        t_x = DistributionOutput.__conform_shape_ndims__(x, ndims=expected_ndims)

        if self.__distribution__.dtype == tf.float32:
            log_prob = self.__distribution__.log_prob(t_x)
        else:
            log_prob = self.__distribution__.log_prob(tf.cast(t_x, self.__distribution__.dtype))

        return DistributionOutput.__conform_shape_ndims__(log_prob)

    def entropy(self):
        if self.__stop_entropy_gradient__:
            ent = tf.stop_gradient(self.__distribution__.entropy())
        else:
            ent = self.__distribution__.entropy()

        return DistributionOutput.__conform_shape_ndims__(ent)

    def sample(self):
        if self.__distribution__.dtype == tf.float32:
            samples = self.__distribution__.sample()
        else:
            samples = tf.cast(self.__distribution__.sample(), tf.float32)

        return DistributionOutput.__conform_shape_ndims__(samples, ndims=2)

    def mode(self):
        if self.__distribution__.dtype == tf.float32:
            samples = self.__distribution__.mode()
        else:
            samples = tf.cast(self.__distribution__.mode(), tf.float32)

        return DistributionOutput.__conform_shape_ndims__(samples, ndims=2)


class PolicyModule(NetworkModule):

    @property
    def num_outputs(self):
        raise NotImplementedError

    @property
    def num_inputs(self):
        raise NotImplementedError


class GroupPolicy(PolicyModule):

    def __init__(self, policies: Union[list, tuple]):
        super().__init__()
        self.__policies__ = policies

    @property
    def weights(self):
        return reduce(lambda x, y: x + y, map(lambda x: x.weights, self.__policies__))

    @property
    def num_outputs(self):
        return reduce(lambda x, y: x + y, map(lambda x: x.num_outputs, self.__policies__))

    @property
    def num_inputs(self):
        return reduce(lambda x, y: x + y, map(lambda x: x.num_inputs, self.__policies__))

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        super().__call__(inputs, *args, **kwargs)
        policies = self.__policies__

        class Output(PolicyOutput):

            def __init__(self):
                input_split = tf.split(inputs, list(map(lambda x: x.num_inputs, policies)), axis=1)
                self.__policy_outputs__ = list(map(lambda x: x[0](x[1]), zip(policies, input_split)))

            def log_prob(self, x: tf.Tensor):
                x_split = tf.split(x, list(map(lambda y: y.num_outputs, policies)), axis=1)
                log_probs = tf.stack(list(map(lambda y: y[0].log_prob(y[1]), zip(self.__policy_outputs__, x_split))),
                                     axis=1)
                log_probs = tf.reduce_sum(log_probs, axis=1)

                return log_probs

            def entropy(self):
                ent = tf.stack(list(map(lambda x: x.entropy(), self.__policy_outputs__)), axis=1)
                ent = tf.reduce_sum(ent, axis=1)

                return ent

            def sample(self):
                sample = tf.concat(list(map(lambda x: x.sample(), self.__policy_outputs__)), axis=1)

                return sample

            def mode(self):
                action = tf.concat(list(map(lambda x: x.mode(), self.__policy_outputs__)), axis=1)

                return action

        return Output()


class BetaPolicy(PolicyModule):

    def __init__(self, num_outputs):
        super().__init__()
        self.__num_outputs__ = num_outputs
        self.__dist__ = None

    @property
    def num_outputs(self):
        return self.__num_outputs__

    @property
    def num_inputs(self):
        return self.__num_outputs__ * 2

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        super().__call__(inputs, *args, **kwargs)
        concentrations = tf.exp(inputs) + 1.0
        concentrations = tf.split(concentrations, 2, axis=1)

        class Output(DistributionOutput):

            def log_prob(self, x: tf.Tensor):
                rounding_value = np.finfo(x.dtype.as_numpy_dtype).tiny
                return super().log_prob(tf.clip_by_value(x, rounding_value, 1.0 - rounding_value))

        return Output(distributions.Beta(*concentrations), stop_entropy_gradient=True)


class BooleanPolicy(PolicyModule):

    def __init__(self, num_outputs):
        super().__init__()
        self.__num_outputs__ = num_outputs

    @property
    def num_outputs(self):
        return self.__num_outputs__

    @property
    def num_inputs(self):
        return self.__num_outputs__

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        super().__call__(inputs, *args, **kwargs)
        return DistributionOutput(distributions.Bernoulli(inputs))


class CategoricalPolicy(PolicyModule):

    def __init__(self, num_categories):
        super().__init__()
        self.__num_categories__ = num_categories
        self.__dist__ = None

    @property
    def num_outputs(self):
        return 1

    @property
    def num_inputs(self):
        return self.__num_categories__

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        super().__call__(inputs, *args, **kwargs)
        return DistributionOutput(distributions.Categorical(inputs))


class GaussianPolicy(PolicyModule):

    def __init__(self, num_outputs):
        super().__init__()
        self.__num_outputs__ = num_outputs
        self.__dist__ = None

    @property
    def num_outputs(self):
        return self.__num_outputs__

    @property
    def num_inputs(self):
        return self.__num_outputs__

    def build(self, input_shape: Union[list, tuple]):
        super().build(input_shape)
        self.__create_weights__(init.zeros()(self.__num_outputs__), 'log_std_params')

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        super().__call__(inputs, *args, **kwargs)
        return DistributionOutput(distributions.MultivariateNormalDiag(inputs,
                                                                       tf.exp(self.__weight_map__['log_std_params'])),
                                  stop_entropy_gradient=True)


class StateNetwork(NetworkModule):
    """StateNetwork is a neural network module that produces a state encoding of an agent's perceived state"""

    def __init__(self, name=None):
        super().__init__(name)
        self.__power_type_encoding_size__ = 8
        self.__weather_encoder__ = ConvolutionEncoder(128,
                                                      name=tf.name_scope("%s%s/" % (self.__scope__.name,
                                                                                    'Weather')).name)
        self.__market_encoder__ = ConvolutionEncoder(128,
                                                     name=tf.name_scope("%s%s/" % (self.__scope__.name,
                                                                                   'Market')).name)
        self.__tariff_encoder__ = DepthEncoder(16, name=self.__scope__.name)
        self.__observation_encoder__ = DenseEncoder(512, name=self.__scope__.name)
        self.__state_encoder__ = tf.keras.layers.LSTM(units=1024, return_sequences=True, return_state=True,
                                                      implementation=2)

    @property
    def weights(self):
        return super().weights + \
               self.__weather_encoder__.weights + \
               self.__market_encoder__.weights + self.__tariff_encoder__.weights + \
               self.__observation_encoder__.weights + self.__state_encoder__.weights

    def state_shape(self, batch_size):
        return batch_size, self.__state_encoder__.units, 2

    def build(self, input_shape: Union[list, tuple]):
        super().build(input_shape)
        k_init = init.orthogonal()

        self.__create_weights__(tf.zeros(self.state_shape(1)), 'initial_state')
        self.__create_weights__(k_init((14, self.__power_type_encoding_size__)), 'power_type_encoding')

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        super().__call__(inputs, *args, **kwargs)

        with self.__operation_scope__:
            input_shape = inputs.shape.as_list()
            state_in = kwargs['state_in']
            state_mask = kwargs['state_mask']

            p_state = tf.where(state_mask,
                               tf.tile(self.__weight_map__['initial_state'], (tf.shape(inputs)[0], 1, 1)),
                               state_in)

            output = tf.reshape(inputs, (-1, input_shape[-1]))
            output = tf.split(output, [7, 125, 168, 26, 460], axis=-1)
            output[1] = self.__weather_encoder__(tf.transpose(tf.reshape(output[1], (-1, 25, 5)), (0, 2, 1)))
            output[2] = self.__market_encoder__(tf.reshape(output[2], (-1, 7, 24)))
            output[4] = self.__tariff_encoder__(self.__encode_power_type__(output[4]))
            output = tf.concat(output, axis=-1)
            output = self.__observation_encoder__(output)
            output = tf.reshape(output, (-1, input_shape[1], output.shape.as_list()[-1]))
            state_encoder_out = self.__state_encoder__(output, initial_state=tf.unstack(p_state, axis=-1))
            output = tf.reshape(state_encoder_out[0], (-1, self.__state_encoder__.units))
            state_out = tf.stack(state_encoder_out[1:], axis=-1)

        return output, state_out

    def __encode_power_type__(self, inputs):
        encoding_size = self.__power_type_encoding_size__
        expected_input_shape = (-1, 20, 23)
        split_inputs = tf.split(tf.reshape(inputs, expected_input_shape), (1, expected_input_shape[-1] - 1), axis=-1)

        power_type_encoding = tf.cast(tf.reshape(split_inputs[0], (-1,)), tf.int32)
        power_type_encoding = tf.nn.embedding_lookup(self.__weight_map__['power_type_encoding'], power_type_encoding)
        power_type_encoding = tf.reshape(power_type_encoding, expected_input_shape[:-1] + (encoding_size,))

        outputs = tf.concat([power_type_encoding, split_inputs[-1]], axis=-1)
        outputs = tf.transpose(outputs, (0, 2, 1))

        return outputs


class ActorNetwork(NetworkModule):
    """StateNetwork is a neural network module that produces the actions of an agent given a state"""

    def __init__(self, name=None):
        super().__init__(name)
        self.__policy__ = GroupPolicy([ActorNetwork.__build_market_policy__(),
                                       GroupPolicy([ActorNetwork.__build_tariff_policy__() for _ in range(5)])])

    @staticmethod
    def __build_market_policy__():
        return GroupPolicy([BooleanPolicy(24), GaussianPolicy(48)])

    @staticmethod
    def __build_tariff_policy__():
        return GroupPolicy([CategoricalPolicy(4), CategoricalPolicy(5), CategoricalPolicy(13), GaussianPolicy(6),
                            BetaPolicy(6), GaussianPolicy(1), BetaPolicy(1), GaussianPolicy(4)])

    @property
    def weights(self):
        return super().weights + self.__policy__.weights

    @property
    def num_outputs(self):
        return self.__policy__.num_outputs

    def build(self, input_shape: Union[list, tuple]):
        super().build(input_shape)

        logit_count = self.__policy__.num_inputs + 1
        k_init = init.variance_scaling()
        b_init = init.zeros()

        self.__create_weights__(k_init((input_shape[-1], logit_count)), 'actor_kernel')
        self.__create_weights__(b_init(logit_count), 'actor_bias')

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        super().__call__(inputs, *args, **kwargs)

        with self.__operation_scope__:
            logits = tf.nn.bias_add(tf.matmul(inputs, self.__weight_map__['actor_kernel']),
                                    self.__weight_map__['actor_bias'])

            logit_split = tf.split(logits, [logits.shape.as_list()[1] - 1, 1], axis=1)
            state_value = tf.squeeze(logit_split[1], axis=1)
            policy = self.__policy__(logit_split[0])

        return policy, state_value


class AgentModule(NetworkModule):

    def __init__(self, name):
        super().__init__(name)
        self.state_network = StateNetwork(name=self.__scope__.name)
        self.actor_network = ActorNetwork(name=self.__scope__.name)

    @property
    def weights(self):
        return self.state_network.weights + self.actor_network.weights

    def __call__(self, inputs: tf.Tensor, *args, **kwargs):
        with self.__operation_scope__:
            hidden_state, output_state = self.state_network(inputs, *args, **kwargs)
            policy, state_value = self.actor_network(hidden_state, *args, **kwargs)

        return policy, state_value, output_state
