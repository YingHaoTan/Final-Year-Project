import server
import struct
import numpy
import tensorflow as tf
import tensorflow.initializers as init
import subprocess
import utility
import queue
import threading
import os
import numpy as np


def __build_internal_state__(scope_name, state_shapes):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        states = tuple([tf.get_variable("state%d" % index, shape=state_shapes[index],
                                        dtype=tf.float32, initializer=init.zeros(), trainable=False)
                        for index in range(len(state_shapes))])

    return states


class AgentServerHook(server.ServerHook):

    def __init__(self, model_builder_fn, num_clients: int, name="AgentServerHook"):
        internal_state = __build_internal_state__(name, model_builder_fn.state_shapes(num_clients))
        model = model_builder_fn(num_clients, internal_state)
        observation_struct = struct.Struct(">%df" % (model.Inputs.shape.dims[-1] + 1))
        output_struct = struct.Struct(">%df" % model.EvaluationOp.shape.dims[-1])

        with tf.control_dependencies([tf.assign(internal_state[idx], model.StateTupleOut[idx])
                                      for idx in range(len(internal_state))]):
            step_op = tf.identity(model.EvaluationOp)

        self.Name = name
        self.InternalState = internal_state
        self.ClientCount = num_clients
        self.Model = model
        self.StepOp = step_op
        self.__observation_struct__ = observation_struct
        self.__output_struct__ = output_struct

    def get_observation_structure(self):
        return self.__observation_struct__

    def get_output_structure(self):
        return self.__output_struct__

    def setup(self, **kwargs):
        pass

    def on_start(self, session, **kwargs):
        session.run([var.initializer for var in self.InternalState])

    def on_step(self, observations, session, **kwargs):
        return session.run(self.StepOp, feed_dict={self.Model.Inputs: numpy.array([observations])[:, :, 1:]})

    def on_stop(self, session, **kwargs):
        pass


class PowerTACGameHook(AgentServerHook):

    def __init__(self, model_builder_fn, num_clients: int, powertac_port: int, name="AgentServerHook"):
        super().__init__(model_builder_fn, num_clients, name)
        self.PowerTACPort = powertac_port
        self.ServerProcess = None
        self.ClientProcesses = [None] * num_clients

    def setup(self, server_instance, **kwargs):
        executable_dir = os.path.join(os.path.dirname(__file__), "bin")

        interpreter_cmd = ["java", "-jar"]
        broker_identities = ",".join("Extreme%d" % idx for idx in range(self.ClientCount))
        self.ServerProcess = subprocess.Popen([*interpreter_cmd,
                                               "server.jar",
                                               "--sim",
                                               "--jms-url",
                                               "tcp://localhost:%d" % self.PowerTACPort,
                                               "--boot-data",
                                               "bootstrap-data",
                                               "--brokers",
                                               broker_identities],
                                              cwd=executable_dir,
                                              stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
        for index in range(self.ClientCount):
            self.ClientProcesses[index] = subprocess.Popen([*interpreter_cmd,
                                                            "broker.jar",
                                                            "--jms-url",
                                                            "tcp://localhost:%d" % self.PowerTACPort,
                                                            "--port",
                                                            str(server_instance.Port),
                                                            "--config",
                                                            "config/broker%d.properties" % index],
                                                           cwd=executable_dir,
                                                           stdin=subprocess.PIPE,
                                                           stdout=subprocess.PIPE,
                                                           stderr=subprocess.PIPE)

    def on_stop(self, session, **kwargs):
        self.ServerProcess.terminate()
        utility.apply(lambda client: client.terminate(), self.ClientProcesses)


class PowerTACRolloutHook(PowerTACGameHook):

    def __init__(self, model_builder_fn, num_clients: int, powertac_port: int,
                 rollout_size: int, recv_queue: queue.Queue, lock: threading.Lock, name="PowerTACRolloutHook"):
        super().__init__(model_builder_fn, num_clients, powertac_port, name)

        self.RolloutSize = rollout_size
        self.RecvQueue = recv_queue
        self.Lock = lock
        self.__rollout_states__ = None
        self.__observation_rollouts__ = numpy.zeros(shape=(rollout_size, num_clients,
                                                           self.Model.Inputs.shape.dims[-1] + 1),
                                                    dtype=numpy.float32)
        self.__action_rollouts__ = numpy.zeros(shape=(rollout_size,
                                                      num_clients,
                                                      self.Model.EvaluationOp.shape.dims[-1]),
                                               dtype=numpy.float32)
        self.__reward_rollouts__ = numpy.zeros(shape=(rollout_size, num_clients), dtype=numpy.float32)
        self.__value_rollouts__ = numpy.zeros(shape=(rollout_size, num_clients), dtype=numpy.float32)
        self.__cash__ = numpy.zeros(shape=(1, num_clients), dtype=numpy.float32)
        self.__rollout_index__ = 0
        self.StepOp = [self.StepOp, tf.squeeze(self.Model.StateValue, axis=-1)]

    def on_start(self, session, **kwargs):
        super().on_start(session, **kwargs)
        self.__rollout_index__ = 0
        self.__cash__.fill(0.0)

    def on_step(self, observations, session, **kwargs):
        if numpy.any(numpy.isnan(observations)):
            print(self.__rollout_index__)
            print(numpy.where(numpy.isnan(observations)))
            print(observations)
            assert False

        self.Lock.acquire()

        rollout_idx = self.__rollout_index__ % self.RolloutSize
        rollout_nidx = rollout_idx + 1
        rollout_pidx = self.RolloutSize - 1 if rollout_idx == 0 else rollout_idx - 1

        if rollout_idx == 0:
            self.__rollout_states__ = numpy.array(session.run(self.InternalState))
        actions, value = super().on_step(observations, session, **kwargs)

        observations = numpy.array([observations])
        value = numpy.array([value])

        cash = observations[:, :, 0]
        self.__reward_rollouts__[rollout_pidx: rollout_pidx + 1, :] = cash - self.__cash__
        self.__cash__ = cash
        if self.__rollout_index__ > 0 and rollout_idx == 0:
            reward_mean = numpy.mean(self.__reward_rollouts__, axis=1, keepdims=True)
            self.__reward_rollouts__ = self.__reward_rollouts__ - reward_mean

            for idx in range(self.ClientCount):
                self.RecvQueue.put((self.__rollout_states__[:, :, idx: idx + 1, :],
                                    self.__observation_rollouts__[:, idx: idx + 1, :],
                                    self.__action_rollouts__[:, idx: idx + 1, :],
                                    self.__reward_rollouts__[:, idx: idx + 1],
                                    self.__value_rollouts__[:, idx: idx + 1],
                                    value[:, idx: idx + 1]))

        self.__observation_rollouts__[rollout_idx: rollout_nidx, :, :] = observations
        self.__action_rollouts__[rollout_idx: rollout_nidx, :, :] = numpy.array([actions])
        self.__value_rollouts__[rollout_idx: rollout_nidx, :] = value
        self.__rollout_index__ = self.__rollout_index__ + 1
        self.Lock.release()

        return actions
