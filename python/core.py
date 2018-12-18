import server
import struct
import numpy
import tensorflow as tf
import tensorflow.initializers as init
import subprocess
import utility
import threading
import queue
import os
from functools import reduce


def __build_internal_state__(scope_name, state_shapes):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        states = tuple([tf.get_variable("state%d" % index, shape=state_shapes[index],
                                        dtype=tf.float32, initializer=init.zeros(), trainable=False)
                        for index in range(len(state_shapes))])

    return states


class Directory:
    EXECUTABLE_DIR = os.path.join(os.path.dirname(__file__), "bin")
    BOOTSTRAP_DIR = os.path.join(EXECUTABLE_DIR, "bootstrap")
    SCRATCH_BOOTSTRAP_DIR = os.path.join(BOOTSTRAP_DIR, "NextGame")


class BootstrapManager:

    def __init__(self, max_parallelism=1):
        self.IsActive = True
        self.SemaphoreMap = {}
        self.ActiveMap = {}
        self.Queue = queue.Queue()
        self.Workers = [threading.Thread(target=self.serve) for _ in range(max_parallelism)]
        utility.apply(lambda worker: worker.start(), self.Workers)

    def serve(self):
        while self.IsActive:
            identifier = self.Queue.get()
            returncode = subprocess.call([*PowerTACGameHook.INTERPRETER_COMMAND,
                                          os.path.relpath(os.path.join(Directory.EXECUTABLE_DIR,
                                                                       "server.jar"),
                                                          Directory.SCRATCH_BOOTSTRAP_DIR),
                                          "--boot",
                                          identifier],
                                         cwd=Directory.SCRATCH_BOOTSTRAP_DIR,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)

            if returncode == 0:
                self.SemaphoreMap[identifier].release()
            else:
                self.Queue.put(identifier)

    def start_bootstrap(self, identifier):
        if identifier not in self.ActiveMap:
            self.ActiveMap[identifier] = False
            self.SemaphoreMap[identifier] = threading.BoundedSemaphore(1)
            self.SemaphoreMap[identifier].acquire()

        if not self.ActiveMap[identifier]:
            self.ActiveMap[identifier] = True
            self.Queue.put(identifier)

    def obtain_bootstrap(self, identifier):
        if identifier not in self.ActiveMap or not self.ActiveMap[identifier]:
            self.start_bootstrap(identifier)

        scratch_bootstrap_file = os.path.join(Directory.SCRATCH_BOOTSTRAP_DIR, identifier)
        bootstrap_file = os.path.join(Directory.BOOTSTRAP_DIR, identifier)
        self.SemaphoreMap[identifier].acquire()

        if os.path.exists(bootstrap_file):
            os.remove(bootstrap_file)

        os.rename(scratch_bootstrap_file, bootstrap_file)
        self.ActiveMap[identifier] = False

        return bootstrap_file

    def kill(self):
        self.IsActive = False
        utility.apply(lambda worker: worker.join(), self.Workers)


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

    def on_reset(self):
        pass


class PowerTACGameHook(AgentServerHook):
    INTERPRETER_COMMAND = ["java", "-jar"]

    def __init__(self, model_builder_fn, num_clients: int, powertac_port: int,
                 cpu_semaphore: threading.BoundedSemaphore,
                 bootstrap_manager: BootstrapManager, name="AgentServerHook"):
        super().__init__(model_builder_fn, num_clients, name)
        self.PowerTACPort = powertac_port
        self.CPUSemaphore = cpu_semaphore
        self.BootstrapManager = bootstrap_manager
        self.SemaphoreAcquired = False
        self.ServerProcess = None
        self.BootstrapProcess = None
        self.ClientProcesses = [None] * num_clients

    def setup(self, server_instance, **kwargs):
        broker_identities = ",".join("Extreme%d" % idx for idx in range(self.ClientCount))
        bootstrap_file = self.BootstrapManager.obtain_bootstrap(self.Name)

        self.CPUSemaphore.acquire()
        print("Starting %s setup" % self.Name)
        self.ServerProcess = subprocess.Popen([*PowerTACGameHook.INTERPRETER_COMMAND,
                                               "server.jar",
                                               "--sim",
                                               "--jms-url",
                                               "tcp://localhost:%d" % self.PowerTACPort,
                                               "--boot-data",
                                               os.path.relpath(bootstrap_file, Directory.EXECUTABLE_DIR),
                                               "--brokers",
                                               broker_identities],
                                              cwd=Directory.EXECUTABLE_DIR,
                                              stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)

        for index in range(self.ClientCount):
            self.ClientProcesses[index] = subprocess.Popen([*PowerTACGameHook.INTERPRETER_COMMAND,
                                                            "broker.jar",
                                                            "--jms-url",
                                                            "tcp://localhost:%d" % self.PowerTACPort,
                                                            "--port",
                                                            str(server_instance.Port),
                                                            "--config",
                                                            "config/broker%d.properties" % index],
                                                           cwd=Directory.EXECUTABLE_DIR,
                                                           stdin=subprocess.PIPE,
                                                           stdout=subprocess.PIPE)

        self.BootstrapManager.start_bootstrap(self.Name)
        self.SemaphoreAcquired = True

    def on_start(self, **kwargs):
        super().on_start(**kwargs)
        self.__try_release_semaphore__()

    def on_step(self, observations, session, **kwargs):
        return super().on_step(observations, session, **kwargs)

    def on_stop(self, session, **kwargs):
        self.__try_release_semaphore__()
        self.ServerProcess.terminate()
        utility.apply(lambda client: client.terminate(), self.ClientProcesses)

    def __try_release_semaphore__(self):
        if self.SemaphoreAcquired:
            self.CPUSemaphore.release()
            self.SemaphoreAcquired = False
            print("%s: Semaphore released" % self.Name)


class PowerTACRolloutHook(PowerTACGameHook):

    def __init__(self, model_builder_fn, num_clients: int, powertac_port: int,
                 cpu_semaphore: threading.BoundedSemaphore,
                 bootstrap_manager: BootstrapManager,
                 rollout_size: int, recv_queue: queue.Queue,
                 alternate_model_name, alternate_policy_prob, gamma_op, lambda_val,
                 name="PowerTACRolloutHook"):
        super().__init__(model_builder_fn, num_clients, powertac_port, cpu_semaphore, bootstrap_manager, name)

        internal_state = __build_internal_state__(name + 'Alt', model_builder_fn.state_shapes(1))
        model = model_builder_fn(self.Model.Inputs[:, -1:, :], internal_state, name=alternate_model_name)

        with tf.control_dependencies([tf.assign(internal_state[idx], model.StateTupleOut[idx])
                                      for idx in range(len(internal_state))]):
            alternate_policy_prob = alternate_policy_prob / (1.0 / num_clients)
            step_op = tf.cond(tf.random_uniform(shape=()) < alternate_policy_prob,
                              lambda: tf.concat([self.StepOp[:-1, :], model.EvaluationOp], axis=0),
                              lambda: self.StepOp)

        self.ModelVersion = 0
        self.ExpectedModelVersion = 0
        self.RolloutSize = rollout_size
        self.RecvQueue = recv_queue
        self.Lambda = lambda_val
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
        self.__log_prob_rollouts__ = numpy.zeros(shape=(rollout_size, num_clients, self.Model.ACTION_COUNT),
                                                 dtype=numpy.float32)
        self.__rollout_index__ = 0
        self.StepOp = [step_op,
                       tf.squeeze(self.Model.StateValue, axis=-1),
                       self.Model.Policies.log_prob(self.StepOp),
                       gamma_op]
        self.AlternateModel = model
        self.AlternateInternalState = internal_state

    def __calculate_reward__(self, cash):
        p_array = numpy.zeros(shape=(1, self.ClientCount), dtype=numpy.float32)
        sorted_indices = numpy.argsort(cash, axis=1)
        last_idx = 0
        for idx in range(self.ClientCount):
            current_cash = cash[0, sorted_indices[0, idx]]
            if idx == self.ClientCount - 1 or cash[0, sorted_indices[0, idx + 1]] > current_cash:
                nidx = idx + 1
                p_value = reduce(lambda a, b: a + b, range(last_idx, nidx)) / (nidx - last_idx)
                for idx_internal in range(last_idx, nidx):
                    p_array[0, sorted_indices[0, idx_internal]] = p_value
                last_idx = nidx
        p_mean = reduce(lambda a, b: a + b, range(self.ClientCount)) / self.ClientCount
        p_variance = reduce(lambda a, b: a + b, map(lambda x: (x - p_mean) ** 2, range(self.ClientCount)))

        return (p_array - p_mean) / numpy.sqrt(p_variance)

    def on_start(self, session, **kwargs):
        super().on_start(session=session, **kwargs)
        self.__rollout_index__ = 0
        session.run([var.initializer for var in self.AlternateInternalState])

    def on_step(self, observations, session, **kwargs):
        if self.ExpectedModelVersion == self.ModelVersion:
            actions = self.__handle_rollout_step__(observations, session, **kwargs)
        else:
            actions, _, _, _ = super().on_step(observations, session, **kwargs)

        return actions

    def update(self):
        self.ModelVersion = self.ModelVersion + 1

    def on_reset(self):
        self.ModelVersion = 0
        self.ExpectedModelVersion = 0

    def __handle_rollout_step__(self, observations, session, **kwargs):
        rollout_idx = self.__rollout_index__ % self.RolloutSize
        rollout_nidx = rollout_idx + 1
        rollout_pidx = self.RolloutSize - 1 if rollout_idx == 0 else rollout_idx - 1

        if rollout_idx == 0:
            self.__rollout_states__ = numpy.array(session.run(self.InternalState))
        actions, value, log_prob, gamma = super().on_step(observations, session, **kwargs)

        observations = numpy.array([observations])
        value = numpy.array([value])

        self.__reward_rollouts__[rollout_pidx: rollout_pidx + 1, :] = self.__calculate_reward__(observations[:, :, 0])
        if self.__rollout_index__ > 0 and rollout_idx == 0:
            print("%s submitting rollout for model update %d" % (self.Name, self.ExpectedModelVersion))
            nvalues = numpy.concatenate([self.__value_rollouts__[1:, :], value], axis=0)

            self.__reward_rollouts__ = self.__reward_rollouts__ * (1 - gamma)
            advantage = self.__reward_rollouts__ + gamma * nvalues - self.__value_rollouts__
            gae_factor = gamma * self.Lambda
            advantage[-2:-1, :] = advantage[-2:-1, :] + gae_factor * advantage[-1:, :]
            for idx in range(advantage.shape[0] - 2):
                gae = gae_factor * advantage[-idx - 2:-idx - 1, :]
                advantage[-idx - 3:-idx - 2, :] = advantage[-idx - 3:-idx - 2, :] + gae

            reward = advantage + self.__value_rollouts__

            for idx in range(self.ClientCount - 1):
                self.RecvQueue.put((self.__rollout_states__[:, :, idx: idx + 1, :],
                                    self.__observation_rollouts__[:, idx: idx + 1, :],
                                    self.__action_rollouts__[:, idx: idx + 1, :],
                                    self.__log_prob_rollouts__[:, idx: idx + 1, :],
                                    advantage[:, idx: idx + 1],
                                    reward[:, idx: idx + 1],
                                    self.__value_rollouts__[:, idx: idx + 1]))
            self.ExpectedModelVersion = self.ExpectedModelVersion + 1

        self.__observation_rollouts__[rollout_idx: rollout_nidx, :, :] = observations[:, :, :]
        self.__action_rollouts__[rollout_idx: rollout_nidx, :, :] = numpy.array([actions])
        self.__log_prob_rollouts__[rollout_idx: rollout_nidx, :, :] = log_prob
        self.__value_rollouts__[rollout_idx: rollout_nidx, :] = value
        self.__rollout_index__ = self.__rollout_index__ + 1

        return actions
