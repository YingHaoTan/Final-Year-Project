import server
import struct
import numpy as np
import tensorflow as tf
import subprocess
import utility
import threading
import queue
import os
import logging
from enum import Enum
from typing import Union, Callable
from functools import reduce
from model import AgentModule
from model import InputNormalizer
from tensorflow import initializers as init


class Directory:
    EXECUTABLE_DIR = os.path.join(os.path.dirname(__file__), "bin")
    BOOTSTRAP_DIR = os.path.join(EXECUTABLE_DIR, "bootstrap")
    SCRATCH_BOOTSTRAP_DIR = os.path.join(BOOTSTRAP_DIR, "scratch")


class ResourceStatus(Enum):
    Wait = 1
    Ready = 2
    Idle = 3


class BootstrapManager:

    def __init__(self, max_parallelism=1):
        self.is_active = True
        self.resource_map = {}
        self.status_map = {}
        self.command_queue = queue.Queue()
        self.workers = [threading.Thread(target=self.serve) for _ in range(max_parallelism)]
        utility.apply(lambda worker: worker.start(), self.workers)

    def serve(self):
        while self.is_active:
            identifier = self.command_queue.get()
            returncode = subprocess.call([*PowerTACGameHook.INTERPRETER_COMMAND,
                                          os.path.relpath(os.path.join(Directory.EXECUTABLE_DIR,
                                                                       "server.jar"),
                                                          Directory.SCRATCH_BOOTSTRAP_DIR),
                                          "--boot",
                                          identifier],
                                         cwd=Directory.SCRATCH_BOOTSTRAP_DIR,
                                         stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)

            if returncode == 0:
                self.status_map[identifier] = ResourceStatus.Ready
                self.resource_map[identifier].put(1)
            else:
                self.command_queue.put(identifier)

    def start_bootstrap(self, identifier):
        if identifier not in self.status_map or self.status_map[identifier] == ResourceStatus.Idle:
            if identifier not in self.resource_map:
                self.resource_map[identifier] = queue.Queue(1)
            self.status_map[identifier] = ResourceStatus.Wait
            self.command_queue.put(identifier)

    def obtain_bootstrap(self, identifier):
        self.start_bootstrap(identifier)
        self.resource_map[identifier].get()

        scratch_bootstrap_file = os.path.join(Directory.SCRATCH_BOOTSTRAP_DIR, identifier)
        bootstrap_file = os.path.join(Directory.BOOTSTRAP_DIR, identifier)

        if os.path.exists(bootstrap_file):
            os.remove(bootstrap_file)

        os.rename(scratch_bootstrap_file, bootstrap_file)
        self.status_map[identifier] = ResourceStatus.Idle

        return bootstrap_file

    def kill(self):
        self.is_active = False
        utility.apply(lambda worker: worker.join(), self.workers)


class AgentServerHook(server.ServerHook):

    def __init__(self, input_normalizer: InputNormalizer, model: AgentModule, num_clients: int, name="AgentServerHook"):
        self.__reset__ = True
        self.name = name
        self.input_normalizer = input_normalizer
        self.model = model
        self.num_clients = num_clients
        self.reset_state_p = tf.placeholder(tf.bool, shape=(1,))
        self.observations_p = tf.placeholder(tf.float32, shape=(num_clients, 1,
                                                                self.input_normalizer.num_inputs))
        self.internal_state_v = tf.Variable(init.zeros()(model.state_network.state_shape(num_clients)), trainable=False)

        self.normalized_observations = tf.reshape(self.observations_p, (-1, self.input_normalizer.num_inputs))
        self.normalized_observations = input_normalizer(self.normalized_observations)
        self.normalized_observations = tf.reshape(self.normalized_observations, tf.shape(self.observations_p))
        policy, state_value, output_state = model(self.normalized_observations, state_in=self.internal_state_v,
                                                  state_mask=tf.tile(self.reset_state_p, (num_clients,)))
        self.policy = policy
        with tf.control_dependencies([tf.assign(self.internal_state_v, output_state)]):
            self.state_value = tf.identity(state_value)
            self.sample_op = policy.sample()
            
    @property
    def observation_structure(self):
        return struct.Struct(">%df" % (self.input_normalizer.num_inputs + 1))

    @property
    def output_structure(self):
        return struct.Struct(">%df" % self.model.actor_network.num_outputs)

    def setup(self, **kwargs):
        pass

    def on_start(self, session, **kwargs):
        self.__reset__ = True

    def on_step(self, observations, **kwargs):
        session = kwargs['session']
        extra_ops = kwargs['extra_ops'] if 'extra_ops' in kwargs else []
        extra_feed_dict = kwargs['extra_feed_dict'] if 'extra_feed_dict' in kwargs else {}

        result = session.run([self.sample_op, self.state_value] + extra_ops,
                             feed_dict={self.observations_p: np.expand_dims(np.array(observations), axis=1)[:, :, 1:],
                                        self.reset_state_p: np.array((self.__reset__,)),
                                        **extra_feed_dict})
        self.__reset__ = False

        return result

    def on_stop(self, session, **kwargs):
        pass

    def on_reset(self):
        pass


class PowerTACGameHook(AgentServerHook):
    INTERPRETER_COMMAND = ["java", "-jar"]

    def __init__(self, input_normalizer: InputNormalizer,  model: AgentModule, num_clients: int, powertac_port: int,
                 cpu_semaphore: threading.BoundedSemaphore,
                 bootstrap_manager: BootstrapManager, name="AgentServerHook"):
        super().__init__(input_normalizer, model, num_clients, name)
        self.powertac_port = powertac_port
        self.semaphore = cpu_semaphore
        self.bootstrap_manager = bootstrap_manager
        self.is_semaphore_acquired = False
        self.server_process = None
        self.client_processes = {}

    def setup(self, server_instance, **kwargs):
        broker_identities = ",".join("Extreme%d" % idx for idx in range(self.num_clients))
        bootstrap_file = self.bootstrap_manager.obtain_bootstrap(self.name)

        self.semaphore.acquire()
        logging.info("Starting %s setup" % self.name)
        self.server_process = subprocess.Popen([*PowerTACGameHook.INTERPRETER_COMMAND,
                                                "server.jar",
                                                "--sim",
                                                "--jms-url",
                                                "tcp://localhost:%d" % self.powertac_port,
                                                "--boot-data",
                                                os.path.relpath(bootstrap_file, Directory.EXECUTABLE_DIR),
                                                "--brokers",
                                                broker_identities],
                                               cwd=Directory.EXECUTABLE_DIR,
                                               stdin=subprocess.DEVNULL,
                                               stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL)

        for index in range(self.num_clients):
            self.client_processes[index] = subprocess.Popen([*PowerTACGameHook.INTERPRETER_COMMAND,
                                                             "broker.jar",
                                                             "--jms-url",
                                                             "tcp://localhost:%d" % self.powertac_port,
                                                             "--port",
                                                             str(server_instance.port),
                                                             "--config",
                                                             "config/broker%d.properties" % index],
                                                            cwd=Directory.EXECUTABLE_DIR,
                                                            stdin=subprocess.DEVNULL,
                                                            stdout=subprocess.DEVNULL,
                                                            stderr=subprocess.DEVNULL)

        self.bootstrap_manager.start_bootstrap(self.name)
        self.is_semaphore_acquired = True

    def on_start(self, **kwargs):
        super().on_start(**kwargs)
        self.__try_release_semaphore__()

    def on_stop(self, session, **kwargs):
        self.__try_release_semaphore__()
        self.server_process.terminate()
        utility.apply(lambda client: client.terminate(), self.client_processes.values())

    def __try_release_semaphore__(self):
        if self.is_semaphore_acquired:
            self.semaphore.release()
            self.is_semaphore_acquired = False
            logging.info("%s: Semaphore released" % self.name)


class PowerTACRolloutHook(PowerTACGameHook):

    def __init__(self, input_normalizer: InputNormalizer,  model: AgentModule, num_clients: int, powertac_port: int,
                 cpu_semaphore: threading.BoundedSemaphore,
                 bootstrap_manager: BootstrapManager,
                 nsteps: int, gamma: Union[Callable[[], float], float], lam: Union[Callable[[], float], float],
                 alt_model: AgentModule,
                 name="PowerTACRolloutHook"):
        super().__init__(input_normalizer, model, num_clients, powertac_port, cpu_semaphore, bootstrap_manager, name)

        self.nsteps = nsteps
        self.rollout_queue = None
        self.gamma = gamma
        self.lam = lam
        self.alt_model = alt_model
        self.alt_internal_state_v = tf.Variable(init.zeros()(self.model.state_network.state_shape(1)),
                                                trainable=False)
        policy, state_value, output_state = alt_model(self.normalized_observations[-1:, :, :],
                                                      state_in=self.alt_internal_state_v,
                                                      state_mask=self.reset_state_p)
        with tf.control_dependencies([tf.assign(self.alt_internal_state_v, output_state)]):
            self.state_value = tf.concat([self.state_value[:-1], state_value], axis=0)
            self.sample_op = tf.concat([self.sample_op[:-1, :], policy.sample()], axis=0)

        self.__states__ = None
        self.__reset_status__ = []
        self.__observation_rollouts__ = np.zeros(shape=(self.num_clients, self.nsteps,
                                                        self.input_normalizer.num_inputs + 1),
                                                 dtype=np.float32)
        self.__action_rollouts__ = np.zeros(shape=(self.num_clients, self.nsteps, self.model.actor_network.num_outputs),
                                            dtype=np.float32)
        self.__reward_rollouts__ = np.zeros(shape=(self.num_clients, self.nsteps), dtype=np.float32)
        self.__value_rollouts__ = np.zeros(shape=(self.num_clients, self.nsteps), dtype=np.float32)
        self.__rollout_index__ = 0

    @staticmethod
    def __evaluate_value__(value: Union[Callable[[], float], float]):
        if callable(value):
            return value()
        else:
            return value

    def __calculate_reward__(self, cash):
        p_array = np.zeros(shape=(self.num_clients, 1), dtype=np.float32)
        sorted_indices = np.argsort(cash, axis=0)
        last_idx = 0

        for idx in range(self.num_clients):
            current_cash = cash[sorted_indices[idx, 0], 0]
            if idx == self.num_clients - 1 or cash[sorted_indices[idx + 1, 0], 0] > current_cash:
                nidx = idx + 1
                p_value = reduce(lambda a, b: a + b, range(last_idx, nidx)) / (nidx - last_idx)
                for idx_internal in range(last_idx, nidx):
                    p_array[sorted_indices[idx_internal, 0], 0] = p_value
                last_idx = nidx
        p_mean = 0.5 * (self.num_clients - 1)
        p_variance = np.mean((np.arange(self.num_clients) - p_mean)**2)

        return (p_array - p_mean) / np.sqrt(p_variance)

    def on_start(self, session, **kwargs):
        super().on_start(session=session, **kwargs)
        self.__rollout_index__ = 0
        self.__reset_status__ = []

    def on_step(self, observations, **kwargs):
        rollout_idx = self.__rollout_index__ % self.nsteps
        rollout_nidx = rollout_idx + 1
        rollout_pidx = self.nsteps - 1 if rollout_idx == 0 else rollout_idx - 1

        if rollout_idx == 0:
            self.__states__ = kwargs['session'].run(self.internal_state_v)
            self.__reset_status__.append(self.__reset__)

        actions, state_values = super().on_step(observations, **kwargs)

        state_values = np.expand_dims(state_values, axis=1)
        observations = np.expand_dims(np.array(observations), axis=1)
        self.__reward_rollouts__[:, rollout_pidx: rollout_pidx + 1] = self.__calculate_reward__(observations[:, :, 0])
        if self.__rollout_index__ > 0 and rollout_idx == 0 and self.rollout_queue is not None:
            nvalues = np.concatenate([self.__value_rollouts__[:, 1:], state_values], axis=1)

            gamma = PowerTACRolloutHook.__evaluate_value__(self.gamma)
            lam = PowerTACRolloutHook.__evaluate_value__(self.lam)

            advantage = self.__reward_rollouts__ + gamma * nvalues - self.__value_rollouts__
            gae_factor = gamma * lam
            advantage[:, -2:-1] = advantage[:, -2:-1] + gae_factor * advantage[:, -1:]
            for idx in range(advantage.shape[1] - 2):
                gae = gae_factor * advantage[:, -idx - 2:-idx - 1]
                advantage[:, -idx - 3:-idx - 2] = advantage[:, -idx - 3:-idx - 2] + gae

            reward = advantage + self.__value_rollouts__

            rollout_reset_status = self.__reset_status__.pop(0)
            assert len(self.__reset_status__) == 1
            for idx in range(self.num_clients - 1):
                self.rollout_queue.put((self.name,
                                        rollout_reset_status,
                                        self.__states__[idx: idx + 1, :, :],
                                        self.__observation_rollouts__[idx: idx + 1, :, :],
                                        self.__action_rollouts__[idx: idx + 1, :, :],
                                        advantage[idx: idx + 1, :],
                                        reward[idx: idx + 1, :],))

        self.__observation_rollouts__[:, rollout_idx: rollout_nidx, :] = observations
        self.__action_rollouts__[:, rollout_idx: rollout_nidx, :] = np.expand_dims(actions, axis=1)
        self.__value_rollouts__[:, rollout_idx: rollout_nidx] = state_values
        self.__rollout_index__ = self.__rollout_index__ + 1

        return actions

    def update_queue(self, rollout_queue: queue.Queue):
        self.rollout_queue = rollout_queue
