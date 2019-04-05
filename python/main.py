import server
import core
import threading
import utility
import queue
import numpy as np
import tensorflow as tf
import os
import logging
import tqdm
import colorama
from model import AgentModule

tf.logging.set_verbosity(tf.logging.ERROR)

BOOTSTRAP_PARALLELISM = 1
SUMMARY_DIR = "summary"
CHECKPOINT_DIR = "ckpt"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "model")
PPO_EPSILON = 0.2
PORT = 60000
PT_PORT = 50000
ROLLOUT_STEPS = 168
BUFFER_SIZE = 648
NUM_MINIBATCH = 18
NUM_EPOCHS = 8
MAX_EPOCHS = 5000 * NUM_EPOCHS
WARMUP_PHASE = 1 * NUM_EPOCHS
MAX_PHASE = 500 * NUM_EPOCHS
LEARNING_RATE = 2e-4

colorama.init()

NUM_CLIENTS = [4] * 14
GAMMA = 0.998
LAMBDA = 0.95
CPU_SEMAPHORE = threading.BoundedSemaphore(BOOTSTRAP_PARALLELISM)


def create_progress():
    return tqdm.tqdm(total=BUFFER_SIZE,
                     bar_format="%s{l_bar}%s{bar}%s{r_bar}" % (colorama.Fore.WHITE,
                                                               colorama.Style.BRIGHT + colorama.Fore.GREEN,
                                                               colorama.Fore.WHITE))


np.warnings.filterwarnings('ignore')
print("Setting up trainer to perform %d training epochs" % MAX_EPOCHS)

logging.basicConfig(filename="main.log", level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")

# Model declaration
model = AgentModule()
base_model = AgentModule()

# Buffer declaration
cash_buffer = np.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS), dtype=np.float32)
reset_buffer = np.zeros(shape=(BUFFER_SIZE,), dtype=np.bool)
states_buffer = np.zeros(shape=model.state_network.state_shape(BUFFER_SIZE), dtype=np.float32)
observation_buffer = np.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS,
                                     model.state_network.num_inputs),
                              dtype=np.float32)
action_buffer = np.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS, model.actor_network.num_outputs),
                         dtype=np.float32)
advantage_buffer = np.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS), dtype=np.float32)
reward_buffer = np.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS), dtype=np.float32)

# Placeholder declaration
stats_ready_variable = tf.Variable(False, trainable=False)
cash_variable = tf.Variable(tf.zeros(cash_buffer.shape, dtype=tf.dtypes.as_dtype(cash_buffer.dtype)),
                            trainable=False)
cash_placeholder = tf.placeholder(cash_variable.dtype, shape=cash_variable.shape)
reset_placeholder = tf.placeholder(tf.dtypes.as_dtype(reset_buffer.dtype), shape=reset_buffer.shape)
states_placeholder = tf.placeholder(tf.dtypes.as_dtype(states_buffer.dtype), shape=states_buffer.shape)
observation_placeholder = tf.placeholder(tf.dtypes.as_dtype(observation_buffer.dtype), shape=observation_buffer.shape)
action_placeholder = tf.placeholder(tf.dtypes.as_dtype(action_buffer.dtype), shape=action_buffer.shape)
advantage_placeholder = tf.placeholder(tf.dtypes.as_dtype(advantage_buffer.dtype), shape=advantage_buffer.shape)
reward_placeholder = tf.placeholder(tf.dtypes.as_dtype(reward_buffer.dtype), shape=reward_buffer.shape)
set_stats_ready_op = tf.assign(stats_ready_variable, tf.convert_to_tensor(True))
set_cash_value_op = tf.assign(cash_variable, cash_placeholder)

# Input pipeline
batch_size = BUFFER_SIZE // NUM_MINIBATCH
dataset = tf.data.Dataset.from_tensor_slices((reset_placeholder, states_placeholder, observation_placeholder,
                                              action_placeholder, advantage_placeholder, reward_placeholder,))
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat(NUM_EPOCHS)
d_iterator = dataset.make_initializable_iterator()
d_reset, d_state, d_obs, d_action, d_adv, d_reward = d_iterator.get_next()
d_action = tf.reshape(d_action, (-1, model.actor_network.num_outputs))
d_adv = tf.reshape(d_adv, (-1, 1))
d_reward = tf.reshape(d_reward, (-1, 1))

# Training operations
policy, state_value, _ = model(d_obs, state_in=d_state, state_mask=d_reset)
base_policy, base_state_value, _ = base_model(d_obs, state_in=d_state, state_mask=d_reset)
prob_ratio = tf.exp(policy.log_prob(d_action) - base_policy.log_prob(d_action))
clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON)
policy_loss = -tf.reduce_mean(tf.minimum(prob_ratio * d_adv, clipped_prob_ratio * d_adv))
clipped_value = base_state_value + tf.clip_by_value(state_value - base_state_value, -PPO_EPSILON, PPO_EPSILON)
value_loss = 0.5 * tf.reduce_mean(tf.maximum((d_reward - clipped_value)**2, (d_reward - state_value)**2))
entropy_loss = 0.01 * tf.reduce_mean(policy.entropy())
loss = policy_loss + value_loss - entropy_loss

global_step = tf.train.create_global_step()
warmup_steps = WARMUP_PHASE * NUM_MINIBATCH
learning_rate = tf.train.polynomial_decay(0.0, global_step, warmup_steps, LEARNING_RATE)
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, centered=True)
grads = tf.gradients(loss, model.trainable_weights)
clipped_grads, global_norm = tf.clip_by_global_norm(grads, 50.0)
gv_list = zip(clipped_grads, model.trainable_weights)
train_op = optimizer.apply_gradients(gv_list, global_step=global_step)

# Tensorboard summary operations
parameter_delta = list(map(lambda x: model.trainable_weights[x] - base_model.trainable_weights[x],
                           range(len(base_model.trainable_weights))))
parameter_delta = tf.concat(list(map(lambda x: tf.reshape(x, [-1]), parameter_delta)), axis=0)
tf.summary.scalar("GlobalNorm", global_norm)
tf.summary.histogram("Cash", cash_variable)
tf.summary.histogram("Advantage", d_adv)
tf.summary.histogram("ParameterDelta", parameter_delta)
with tf.name_scope("Losses"):
    tf.summary.scalar("Policy", policy_loss)
    tf.summary.scalar("Value", value_loss)
    tf.summary.scalar("Entropy", entropy_loss)
    tf.summary.scalar("Combined", loss)
with tf.name_scope("Gradients"):
    tf.summary.histogram("Clipped", tf.concat([tf.reshape(grad, [-1]) for grad in clipped_grads], axis=0))
with tf.name_scope("Reward"):
    tf.summary.histogram("Actual", d_reward)
    tf.summary.histogram("Trained Prediction", state_value)
    tf.summary.histogram("Base Prediction", base_state_value)
with tf.name_scope("ProbabilityRatio"):
    tf.summary.histogram("Clipped", clipped_prob_ratio)
    tf.summary.histogram("Unclipped", prob_ratio)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(SUMMARY_DIR, graph=tf.get_default_graph())

# Server declaration
bootstrap_manager = core.BootstrapManager(BOOTSTRAP_PARALLELISM)
servers = [server.Server(NUM_CLIENTS[idx], PORT + idx,
                         core.PowerTACRolloutHook(model, NUM_CLIENTS[idx], PT_PORT + idx,
                                                  CPU_SEMAPHORE, bootstrap_manager, ROLLOUT_STEPS,
                                                  GAMMA, LAMBDA, base_model, name="Game%02d" % idx))
           for idx in range(len(NUM_CLIENTS))]

# Training session region
variable_list = [global_step, cash_variable, stats_ready_variable] + \
                model.weights + \
                base_model.weights + \
                optimizer.variables() + \
                list(map(lambda x: x.hook.internal_state_v, servers)) + \
                list(map(lambda x: x.hook.alt_internal_state_v, servers))
sync_op = tf.group([tf.assign(dest, src) for src, dest in zip(model.weights, base_model.weights)])
saver = tf.train.Saver(var_list=variable_list)

sess = tf.Session()
if tf.train.latest_checkpoint(CHECKPOINT_DIR) is None:
    sess.run(tf.initializers.variables(variable_list))
    sess.run(sync_op)
    print("Initialized variables for training session")
else:
    sess.run(tf.initializers.variables(variable_list))
    saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_DIR))
    print("Restored variables from previous training session")

server_threads = [threading.Thread(target=server.serve, kwargs={"session": sess}) for server in servers]
utility.apply(lambda thread: thread.start(), server_threads)

stats_ready, step_count = sess.run([stats_ready_variable, global_step])

rollout_queue = queue.Queue()
rollout_idx = 0
update_idx = step_count // (NUM_EPOCHS * NUM_MINIBATCH)
progress = None
print("Starting training session from update %d" % (step_count // (NUM_MINIBATCH * NUM_EPOCHS)))
while rollout_queue is not None:
    utility.apply(lambda hook: hook.update_queue(rollout_queue), map(lambda s: s.hook, servers))

    for (name, reset, states, observations, actions, advantages, rewards) in iter(rollout_queue.get, None):
        progress = create_progress() if progress is None else progress

        rollout_nidx = rollout_idx + 1
        cash_buffer[rollout_idx: rollout_nidx, :] = observations[:, :, 0]
        reset_buffer[rollout_idx] = reset
        states_buffer[rollout_idx: rollout_nidx, :] = states
        observation_buffer[rollout_idx: rollout_nidx, :] = observations[:, :, 1:]
        action_buffer[rollout_idx: rollout_nidx, :, :] = actions
        advantage_buffer[rollout_idx: rollout_nidx, :] = advantages
        reward_buffer[rollout_idx: rollout_nidx, :] = rewards

        progress.update()
        logging.info("Received rollout %d for update %d from %s" % (rollout_idx, update_idx, name))
        rollout_idx = rollout_nidx % BUFFER_SIZE
        if rollout_idx == 0:
            sess.run(sync_op)

            with CPU_SEMAPHORE:
                if stats_ready:
                    sess.run([d_iterator.initializer, set_cash_value_op],
                             feed_dict={cash_placeholder: cash_buffer,
                                        reset_placeholder: reset_buffer,
                                        states_placeholder: states_buffer,
                                        observation_placeholder: observation_buffer,
                                        action_placeholder: action_buffer,
                                        advantage_placeholder: advantage_buffer,
                                        reward_placeholder: reward_buffer})

                    summary_value = None
                    while True:
                        try:
                            _, summary_value, step_count = sess.run([train_op, summary_op, global_step])
                            logging.info("Updating agent parameters step %d" % step_count)
                        except tf.errors.OutOfRangeError:
                            saver.save(sess, CHECKPOINT_PREFIX, global_step=step_count)
                            summary_writer.add_summary(summary_value, global_step=step_count)
                            update_idx = step_count // (NUM_EPOCHS * NUM_MINIBATCH)
                            break

                observation_stats = np.reshape(observation_buffer, (-1, model.state_network.num_inputs))
                model.state_network.update_input_statistics(sess,
                                                            mean=observation_stats.mean(axis=0, keepdims=True),
                                                            var=observation_stats.var(axis=0, keepdims=True),
                                                            count=observation_stats.shape[0])
                if not stats_ready:
                    progress.set_postfix_str("Initial statistics gathered, commencing training session")
                    sess.run([sync_op, set_stats_ready_op])
                    utility.apply(lambda server: server.reset(), servers)
                else:
                    progress.set_postfix_str("Update %d completed" % update_idx)

                progress.close()
                rollout_queue, stats_ready, progress = queue.Queue(), True, None
                break

    if step_count > MAX_EPOCHS * NUM_MINIBATCH:
        print("Completed training process, terminating session")
        utility.apply(lambda server: server.stop(), servers)
        rollout_queue = None

utility.apply(lambda thread: thread.join(), server_threads)
bootstrap_manager.kill()
