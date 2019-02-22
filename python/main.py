import server
import model
import core
import threading
import utility
import queue
import numpy
import tensorflow as tf
import os
import logging
import tqdm

CLIENT_PARALLELISM = 1
SUMMARY_DIR = "summary"
CHECKPOINT_DIR = "ckpt"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "model")
PPO_EPSILON = 0.2
PORT = 60000
PT_PORT = 50000
ROLLOUT_STEPS = 168
NBATCHES = 15
BUFFER_SIZE = 630
NUM_MINIBATCH = 21
NUM_EPOCHS = 16
MAX_EPOCHS = 2500 * NUM_EPOCHS
WARMUP_PHASE = 1 * NUM_EPOCHS
MAX_PHASE = 250 * NUM_EPOCHS
INITIAL_LR = 0.0002
FINAL_LR = 0.00002

NUM_CLIENTS = [4] * 14
LAMBDA = 0.95
ROLLOUT_QUEUE = queue.Queue()
CPU_SEMAPHORE = threading.BoundedSemaphore(CLIENT_PARALLELISM)

numpy.warnings.filterwarnings('ignore')
print("Setting up trainer to perform %d training epochs" % MAX_EPOCHS)

logging.basicConfig(filename="main.log", level=logging.INFO,
                    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s")

reset_placeholder = tf.placeholder(tf.bool, BUFFER_SIZE)
state_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, model.Model.HIDDEN_STATE_COUNT))
obs_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS, model.Model.FEATURE_COUNT + 1))
action_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS, model.Model.ACTION_COUNT))
log_prob_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS))
reward_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS))
advantage_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS))
value_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS))

dataset = tf.data.Dataset.from_tensor_slices((reset_placeholder, state_placeholder,
                                              obs_placeholder, action_placeholder, log_prob_placeholder,
                                              reward_placeholder, advantage_placeholder, value_placeholder))
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BUFFER_SIZE // NUM_MINIBATCH)
dataset = dataset.repeat(NUM_EPOCHS)
d_iterator = dataset.make_initializable_iterator()

d_reset, d_state, d_obs, d_action, d_log_prob, d_reward, d_adv, d_value = d_iterator.get_next()
d_state = tf.reshape(d_state, (BUFFER_SIZE // NUM_MINIBATCH, model.Model.HIDDEN_STATE_COUNT))
d_obs = tf.transpose(d_obs, (1, 0, 2))
d_action = tf.reshape(tf.transpose(d_action, (1, 0, 2)), (-1, model.Model.ACTION_COUNT))
d_log_prob = tf.reshape(tf.transpose(d_log_prob, (1, 0)), [-1])
d_reward = tf.reshape(tf.transpose(d_reward, (1, 0)), [-1])
d_adv = tf.reshape(tf.transpose(d_adv, (1, 0)), [-1])
d_value = tf.reshape(tf.transpose(d_value, (1, 0)), [-1])

dataset = tf.data.Dataset.from_tensor_slices((obs_placeholder, advantage_placeholder))
dataset = dataset.map(lambda x, y: (x[:, 0], y))
dataset = dataset.batch(BUFFER_SIZE)
dataset = dataset.repeat(NUM_EPOCHS * NUM_MINIBATCH)
d_summaryiterator = dataset.make_initializable_iterator()
d_cash, d_sadv = d_summaryiterator.get_next()

alt_model = model.Model(d_obs[:, :, 1:], d_state, d_reset, name='AlternateModel')
cmodel = model.Model(d_obs[:, :, 1:], d_state, d_reset)

transfer_op = model.Model.create_transfer_op(cmodel, alt_model)
global_step = tf.train.create_global_step()
gamma_shift_op = tf.cast(tf.minimum(global_step / (MAX_PHASE * NUM_MINIBATCH), 1.0), tf.float32)
gamma_op = 0.998 + 0.0017 * gamma_shift_op

d_adv = tf.expand_dims(d_adv, axis=1)
prob_ratio = tf.exp(cmodel.Policies.log_prob(d_action) - d_log_prob)
clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON)
policy_loss = -tf.reduce_mean(tf.minimum(prob_ratio * d_adv, clipped_prob_ratio * d_adv))
value_prediction = tf.squeeze(cmodel.StateValue, axis=1)
value_clipped = d_value + tf.clip_by_value(value_prediction - d_value, -PPO_EPSILON, PPO_EPSILON)
value_clipped_loss = (value_clipped - d_reward) ** 2
value_unclipped_loss = (value_prediction - d_reward) ** 2
value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_clipped, value_unclipped_loss))
entropy_loss = 0.01 * tf.reduce_mean(cmodel.Policies.entropy())
loss = policy_loss + value_loss - entropy_loss

warmup_steps = WARMUP_PHASE * NUM_MINIBATCH
steady_steps = MAX_PHASE * NUM_MINIBATCH
warmup_lr = tf.train.polynomial_decay(0.0, global_step, warmup_steps, INITIAL_LR)
lr_decay = tf.train.polynomial_decay(INITIAL_LR, global_step - steady_steps, MAX_EPOCHS * NUM_MINIBATCH, FINAL_LR)
lr = tf.case([(tf.less_equal(global_step, warmup_steps), lambda: warmup_lr),
              (tf.greater_equal(global_step, steady_steps), lambda: lr_decay)],
             default=lambda: tf.convert_to_tensor(INITIAL_LR))
optimizer = tf.train.RMSPropOptimizer(lr, decay=0.99, centered=True)
grads = tf.gradients(loss, cmodel.variables)
clipped_grads, global_norm = tf.clip_by_global_norm(grads, 50.0)
grads_n_vars = zip(clipped_grads, cmodel.variables)
train_op = optimizer.apply_gradients(grads_n_vars, global_step=global_step)
parameter_delta = [cmodel.variables[idx] - alt_model.variables[idx] for idx in range(len(cmodel.variables))]

tf.summary.scalar("GlobalNorm", global_norm)
tf.summary.histogram("Cash", d_cash)
tf.summary.histogram("Embedding", cmodel.Embedding)
tf.summary.histogram("Advantage", d_sadv)
tf.summary.histogram("ParameterDelta", tf.concat([tf.reshape(pdelta, [-1]) for pdelta in parameter_delta], axis=0))
with tf.name_scope("Losses"):
    tf.summary.scalar("Policy", policy_loss)
    tf.summary.scalar("Value", value_loss)
    tf.summary.scalar("Entropy", entropy_loss)
    tf.summary.scalar("Combined", loss)
with tf.name_scope("Gradients"):
    tf.summary.histogram("Clipped", tf.concat([tf.reshape(grad, [-1]) for grad in clipped_grads], axis=0))
with tf.name_scope("Reward"):
    tf.summary.histogram("Actual", d_reward)
    tf.summary.histogram("Predicted", cmodel.StateValue)
with tf.name_scope("ProbabilityRatio"):
    tf.summary.histogram("Clipped", clipped_prob_ratio)
    tf.summary.histogram("Unclipped", prob_ratio)
with tf.name_scope("Value"):
    tf.summary.histogram("Clipped", tf.clip_by_value(value_prediction - d_value, -PPO_EPSILON, PPO_EPSILON))
    tf.summary.histogram("Unclipped", value_prediction - d_value)

summary_op = tf.summary.merge_all()
saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(SUMMARY_DIR, graph=tf.get_default_graph())

bootstrap_manager = core.BootstrapManager(CLIENT_PARALLELISM)
servers = [server.Server(NUM_CLIENTS[idx], PORT + idx,
                         core.PowerTACRolloutHook(model.Model, NUM_CLIENTS[idx], PT_PORT + idx,
                                                  CPU_SEMAPHORE, bootstrap_manager, ROLLOUT_STEPS, NBATCHES,
                                                  ROLLOUT_QUEUE, alt_model.Name, 0.2, gamma_op, LAMBDA,
                                                  name="Game%02d" % idx))
           for idx in range(len(NUM_CLIENTS))]

sess = tf.Session()

sess.run(tf.global_variables_initializer())
if tf.train.latest_checkpoint(CHECKPOINT_DIR) is not None:
    saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_DIR))
else:
    sess.run(transfer_op)

server_threads = [threading.Thread(target=server.serve, kwargs={"session": sess}) for server in servers]
utility.apply(lambda thread: thread.start(), server_threads)

reset_buffer = numpy.zeros(shape=BUFFER_SIZE, dtype=numpy.bool)
state_buffer = numpy.zeros(shape=(BUFFER_SIZE, model.Model.HIDDEN_STATE_COUNT),
                           dtype=numpy.float32)
observation_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS, model.Model.FEATURE_COUNT + 1),
                                 dtype=numpy.float32)
action_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS, model.Model.ACTION_COUNT),
                            dtype=numpy.float32)
log_prob_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS), dtype=numpy.float32)
advantage_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS), dtype=numpy.float32)
reward_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS), dtype=numpy.float32)
value_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS), dtype=numpy.float32)
rollout_idx = 0
step_count = sess.run(global_step)

print("Starting training session\n")
progress = tqdm.tqdm(total=BUFFER_SIZE, 
                     bar_format="%s{l_bar}%s{bar}%s{r_bar}" % (colorama.Fore.WHITE, 
                                                               colorama.Style.BRIGHT + colorama.Fore.GREEN, 
                                                               colorama.Fore.WHITE))
for rollout in iter(ROLLOUT_QUEUE.get, None):
    reset_buffer[rollout_idx] = rollout[0]
    state_buffer[rollout_idx: rollout_idx + 1, :] = rollout[1]
    observation_buffer[rollout_idx: rollout_idx + 1, :, :] = numpy.transpose(rollout[2], (1, 0, 2))
    action_buffer[rollout_idx: rollout_idx + 1, :, :] = numpy.transpose(rollout[3], (1, 0, 2))
    log_prob_buffer[rollout_idx: rollout_idx + 1, :] = numpy.transpose(rollout[4], (1, 0))
    advantage_buffer[rollout_idx: rollout_idx + 1, :] = numpy.transpose(rollout[5], (1, 0))
    reward_buffer[rollout_idx: rollout_idx + 1, :] = numpy.transpose(rollout[6], (1, 0))
    value_buffer[rollout_idx: rollout_idx + 1, :] = numpy.transpose(rollout[7], (1, 0))

    rollout_idx = (rollout_idx + 1) % BUFFER_SIZE
    progress.update()
    if rollout_idx == 0:
        sess.run(transfer_op)
        stats_count = sess.run(cmodel.RunningStats.Count)

        with CPU_SEMAPHORE:
            if stats_count > 0:
                sess.run([d_iterator.initializer, d_summaryiterator.initializer],
                         feed_dict={reset_placeholder: reset_buffer,
                                    state_placeholder: state_buffer,
                                    obs_placeholder: observation_buffer,
                                    action_placeholder: action_buffer,
                                    log_prob_placeholder: log_prob_buffer,
                                    advantage_placeholder: advantage_buffer,
                                    reward_placeholder: reward_buffer,
                                    value_placeholder: value_buffer})
                summary_value = None
                data_present = True
                while data_present:
                    try:
                        _, summary_value, step_count = sess.run([train_op, summary_op, global_step])
                    except tf.errors.OutOfRangeError:
                        saver.save(sess, CHECKPOINT_PREFIX, global_step=step_count)
                        summary_writer.add_summary(summary_value, global_step=step_count)

                        update_idx = step_count // (NUM_EPOCHS * NUM_MINIBATCH)
                        data_present = False

            stats_obs_buffer = numpy.reshape(observation_buffer[:, :, 1:], (-1, model.Model.FEATURE_COUNT))
            sess.run(cmodel.RunningStats.UpdateStatsOp,
                     feed_dict={cmodel.RunningStats.MeanInput: stats_obs_buffer.mean(axis=0, keepdims=True),
                                cmodel.RunningStats.VarInput: stats_obs_buffer.var(axis=0, keepdims=True),
                                cmodel.RunningStats.CountInput: stats_obs_buffer.shape[0]})

            if stats_count == 0:
                progress.set_postfix_str("Initializing input statistics with %d data points" % stats_count)
                utility.apply(lambda server: server.reset(), servers)
                with ROLLOUT_QUEUE.mutex:
                    ROLLOUT_QUEUE.queue.clear()
            else:
                progress.set_postfix_str("Update %d completed, %d input statistics collected" % (update_idx, stats_count))
                utility.apply(lambda hook: hook.update(), map(lambda server: server.Hook, servers))
        if step_count > MAX_EPOCHS * NUM_MINIBATCH:
            progress.set_postfix_str("Completed training process, terminating session")
            utility.apply(lambda server: server.stop(), servers)
            break
            
        progress.close()
        progress = tqdm.tqdm(total=BUFFER_SIZE,
                            bar_format="%s{l_bar}%s{bar}%s{r_bar}" % (colorama.Fore.WHITE, 
                                                               colorama.Style.BRIGHT + colorama.Fore.GREEN, 
                                                               colorama.Fore.WHITE))

utility.apply(lambda thread: thread.join(), server_threads)
bootstrap_manager.kill()
