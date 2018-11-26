import server
import model
import core
import threading
import utility
import queue
import numpy
import tensorflow as tf
import os
import random

SUMMARY_DIR = "summary"
CHECKPOINT_DIR = "ckpt"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "model")
PPO_EPSILON = 0.2
PORT = 61000
PT_PORT = 60000
ROLLOUT_STEPS = 168
BUFFER_SIZE = 36
NUM_MINIBATCH = 2
NUM_EPOCHS = 8
MAX_EPOCHS = 5000 * NUM_EPOCHS
MAX_PHASE = 2500 * NUM_EPOCHS
INITIAL_PHASE = 500 * NUM_EPOCHS
INITIAL_LR = 1e-4
FINAL_LR = 1e-5
PRD_C = 0.015

NUM_CLIENTS = [4] * 12
LAMBDA = 0.95
ROLLOUT_QUEUE = queue.Queue()
CPU_SEMAPHORE = threading.BoundedSemaphore(1)


print("Setting up trainer to perform %d training epochs" % MAX_EPOCHS)

state_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, 2, 1, model.Model.HIDDEN_STATE_COUNT))
obs_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS, model.Model.FEATURE_COUNT + 1))
action_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS, model.Model.ACTION_COUNT))
log_prob_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS, model.Model.ACTION_COUNT))
reward_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS))
advantage_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS))

dataset = tf.data.Dataset.from_tensor_slices((state_placeholder,
                                              obs_placeholder, action_placeholder, log_prob_placeholder,
                                              reward_placeholder, advantage_placeholder))
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BUFFER_SIZE // NUM_MINIBATCH)
dataset = dataset.repeat(NUM_EPOCHS)
d_iterator = dataset.make_initializable_iterator()

d_state, d_obs, d_action, d_log_prob, d_reward, d_adv = d_iterator.get_next()
d_state = tf.transpose(d_state, (1, 2, 0, 3))
d_obs = tf.transpose(d_obs, (1, 0, 2))
d_action = tf.reshape(tf.transpose(d_action, (1, 0, 2)), (-1, model.Model.ACTION_COUNT))
d_log_prob = tf.reshape(tf.transpose(d_log_prob, (1, 0, 2)), [-1, model.Model.ACTION_COUNT])
d_reward = tf.reshape(tf.transpose(d_reward, (1, 0)), [-1])
d_adv = tf.reshape(tf.transpose(d_adv, (1, 0)), [-1])

dataset = tf.data.Dataset.from_tensor_slices((obs_placeholder, advantage_placeholder))
dataset = dataset.map(lambda x, y: (x[:, 0], y))
dataset = dataset.batch(BUFFER_SIZE)
dataset = dataset.repeat(NUM_EPOCHS * NUM_MINIBATCH)
d_summaryiterator = dataset.make_initializable_iterator()
d_cash, d_sadv = d_summaryiterator.get_next()

alt_model = model.Model(d_obs[:, :, 1:], tf.unstack(d_state, 2), name='AlternateModel')
cmodel = model.Model(d_obs[:, :, 1:], tf.unstack(d_state, 2))

transfer_op = model.Model.create_transfer_op(cmodel, alt_model)
global_step = tf.train.create_global_step()
objective_step = tf.maximum(global_step - (INITIAL_PHASE * NUM_MINIBATCH), 0)
objective_shift_op = tf.cast(tf.minimum(objective_step / (MAX_PHASE * NUM_MINIBATCH), 1.0), tf.float32)
gamma_shift_op = tf.cast(tf.minimum(global_step / (MAX_PHASE * NUM_MINIBATCH), 1.0), tf.float32)
gamma_op = 0.998 + 0.0017 * gamma_shift_op

d_adv = tf.expand_dims(d_adv, axis=1)
prob_ratio = tf.exp(tf.reduce_sum(cmodel.Policies.log_prob(d_action) - d_log_prob, axis=-1))
clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON)
policy_loss = -tf.reduce_mean(tf.minimum(prob_ratio * d_adv,
                                         clipped_prob_ratio * d_adv))
value_prediction = tf.squeeze(cmodel.StateValue, axis=1)
value_loss = 0.5 * tf.reduce_mean((value_prediction - d_reward) ** 2)
entropy_loss = 0.005 * tf.reduce_mean(cmodel.Policies.entropy())
loss = policy_loss + value_loss - entropy_loss

warmup_steps = NUM_MINIBATCH * NUM_EPOCHS
warmup_lr = tf.train.polynomial_decay(0.0, global_step, warmup_steps, INITIAL_LR)
lr = tf.train.polynomial_decay(INITIAL_LR, global_step - warmup_steps, MAX_EPOCHS * NUM_MINIBATCH, FINAL_LR)
lr = tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
optimizer = tf.train.RMSPropOptimizer(lr, decay=0.99, centered=True)
grads = tf.gradients(loss, cmodel.variables)
clipped_grads, global_norm = tf.clip_by_global_norm([tf.identity(grad) for grad in grads], 30.0)
grads_n_vars = zip(clipped_grads, cmodel.variables)
train_op = optimizer.apply_gradients(grads_n_vars, global_step=global_step)

tf.summary.scalar("Advantages", tf.reduce_mean(d_sadv))
tf.summary.scalar("GNorm", global_norm)
tf.summary.histogram("Cash", d_cash)
tf.summary.histogram("Embedding", cmodel.Embedding)
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
    tf.summary.scalar("MaximumCash", tf.reduce_max(d_cash))
with tf.name_scope("ProbabilityRatio"):
    tf.summary.histogram("Clipped", clipped_prob_ratio)


summary_op = tf.summary.merge_all()

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(SUMMARY_DIR, graph=tf.get_default_graph())

servers = [server.Server(NUM_CLIENTS[idx], PORT + idx,
                         core.PowerTACRolloutHook(model.Model, NUM_CLIENTS[idx], PT_PORT + idx,
                                                  CPU_SEMAPHORE, ROLLOUT_STEPS, ROLLOUT_QUEUE,
                                                  alt_model.Name, 0.2, objective_shift_op, gamma_op, LAMBDA,
                                                  name="Game%02d" % idx))
           for idx in range(len(NUM_CLIENTS))]

sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(transfer_op)

server_threads = [threading.Thread(target=server.serve, kwargs={"session": sess}) for server in servers]
utility.apply(lambda thread: thread.start(), server_threads)

state_buffer = numpy.zeros(shape=(BUFFER_SIZE, 2, 1, model.Model.HIDDEN_STATE_COUNT),
                           dtype=numpy.float32)
observation_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS, model.Model.FEATURE_COUNT + 1),
                                 dtype=numpy.float32)
action_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS, model.Model.ACTION_COUNT),
                            dtype=numpy.float32)
log_prob_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS, model.Model.ACTION_COUNT), dtype=numpy.float32)
reward_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS), dtype=numpy.float32)
value_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS), dtype=numpy.float32)
rollout_idx = 0
step_count = sess.run(global_step)

prd_counter = 1
for rollout in iter(ROLLOUT_QUEUE.get, None):
    state_buffer[rollout_idx: rollout_idx + 1, :, :, :] = numpy.transpose(rollout[0], (2, 0, 1, 3))
    observation_buffer[rollout_idx: rollout_idx + 1, :, :] = numpy.transpose(rollout[1], (1, 0, 2))
    action_buffer[rollout_idx: rollout_idx + 1, :, :] = numpy.transpose(rollout[2], (1, 0, 2))
    log_prob_buffer[rollout_idx: rollout_idx + 1, :] = numpy.transpose(rollout[3], (1, 0, 2))
    reward_buffer[rollout_idx: rollout_idx + 1, :] = numpy.transpose(rollout[4], (1, 0))
    value_buffer[rollout_idx: rollout_idx + 1, :] = numpy.transpose(rollout[5], (1, 0))

    rollout_idx = (rollout_idx + 1) % BUFFER_SIZE
    if rollout_idx == 0:
        stats_count = sess.run(cmodel.RunningStats.Count)

        print("Running Statistics Count: %d" % stats_count)
        with CPU_SEMAPHORE:
            if stats_count == 0:
                utility.apply(lambda server: server.reset(), servers)
            else:
                sess.run([d_iterator.initializer, d_summaryiterator.initializer],
                         feed_dict={state_placeholder: state_buffer,
                                    obs_placeholder: observation_buffer,
                                    action_placeholder: action_buffer,
                                    log_prob_placeholder: log_prob_buffer,
                                    reward_placeholder: value_buffer,
                                    advantage_placeholder: reward_buffer})
                summary_value = None
                data_present = True
                mode_prob = 0
                while data_present:
                    try:
                        _, summary_value, step_count = sess.run([train_op, summary_op, global_step])
                    except tf.errors.OutOfRangeError:
                        saver.save(sess, CHECKPOINT_PREFIX, global_step=step_count)
                        summary_writer.add_summary(summary_value, global_step=step_count)

                        update_idx = step_count // (NUM_EPOCHS * NUM_MINIBATCH)
                        print("Update %d completed" % update_idx)

                        data_present = False

            stats_obs_buffer = numpy.reshape(observation_buffer[:, :, 1:], (-1, model.Model.FEATURE_COUNT))
            sess.run(cmodel.RunningStats.UpdateStatsOp,
                     feed_dict={cmodel.RunningStats.MeanInput: stats_obs_buffer.mean(axis=0, keepdims=True),
                                cmodel.RunningStats.VarInput: stats_obs_buffer.var(axis=0, keepdims=True),
                                cmodel.RunningStats.CountInput: stats_obs_buffer.shape[0]})

            if stats_count > 0:
                utility.apply(lambda hook: hook.update(), map(lambda server: server.Hook, servers))
                if random.random() < PRD_C * prd_counter:
                    sess.run(transfer_op)
                    prd_counter = 1
                    print("Transferred parameters to alternate model")
                else:
                    prd_counter = prd_counter + 1
        if step_count > MAX_EPOCHS * NUM_MINIBATCH:
            print("Completed training process, terminating session")
            utility.apply(lambda server: server.stop(), servers)
            break

utility.apply(lambda thread: thread.join(), server_threads)
