import server
import model
import core
import threading
import utility
import queue
import numpy
import tensorflow as tf
import scipy.signal as signal


SUMMARY_DIR = "summary"
CHECKPOINT_PREFIX = "ckpt/model"
PPO_EPSILON = 0.2
PORT = 61000
PT_PORT = 60000
ROLLOUT_STEPS = 24
BUFFER_SIZE = 48
NUM_MINIBATCH = 3
NUM_EPOCHS = 4
MAX_EPOCHS = 10000 * NUM_EPOCHS
INITIAL_LR = 3e-4
FINAL_LR = 3e-5

NUM_CLIENTS = [4] * 12
LAMBDA = 0.95
ROLLOUT_QUEUE = queue.Queue()
CPU_SEMAPHORE = threading.BoundedSemaphore(1)


def gamma(step):
    return 0.99 + 0.09 * (step / (MAX_EPOCHS * NUM_MINIBATCH))


print("Setting up trainer to perform %d training epochs" % MAX_EPOCHS)

state_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, 2, 1, model.Model.HIDDEN_STATE_COUNT))
obs_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS, model.Model.FEATURE_COUNT + 1))
action_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS, model.Model.ACTION_COUNT))
reward_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS))
advantage_placeholder = tf.placeholder(tf.float32, (BUFFER_SIZE, ROLLOUT_STEPS))

dataset = tf.data.Dataset.from_tensor_slices((state_placeholder,
                                              obs_placeholder, action_placeholder,
                                              reward_placeholder, advantage_placeholder))
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BUFFER_SIZE // NUM_MINIBATCH)
dataset = dataset.repeat(NUM_EPOCHS)
d_iterator = dataset.make_initializable_iterator()

d_state, d_obs, d_action, d_reward, d_adv = d_iterator.get_next()
d_state = tf.transpose(d_state, (1, 2, 0, 3))
d_obs = tf.transpose(d_obs, (1, 0, 2))
d_action = tf.reshape(tf.transpose(d_action, (1, 0, 2)), (-1, model.Model.ACTION_COUNT))
d_reward = tf.reshape(tf.transpose(d_reward, (1, 0)), [-1])
d_adv = tf.reshape(tf.transpose(d_adv, (1, 0)), [-1])

dataset = tf.data.Dataset.from_tensor_slices((obs_placeholder, advantage_placeholder))
dataset = dataset.map(lambda x, y: (x[:, 0], y))
dataset = dataset.batch(BUFFER_SIZE)
dataset = dataset.repeat(NUM_EPOCHS * NUM_MINIBATCH)
d_summaryiterator = dataset.make_initializable_iterator()
d_cash, d_sadv = d_summaryiterator.get_next()

omodel = model.Model(d_obs[:, :, 1:], (d_state[0], d_state[1]), name="OModel")
cmodel = model.Model(d_obs[:, :, 1:], (d_state[0], d_state[1]))

global_step = tf.train.create_global_step()
d_adv = tf.expand_dims(d_adv, axis=1)
prob_ratio = tf.exp(omodel.Policies.neglogp(d_action) - cmodel.Policies.neglogp(d_action))
policy_loss = -tf.reduce_mean(tf.reduce_sum(tf.minimum(prob_ratio * d_adv,
                                                       tf.clip_by_value(prob_ratio,
                                                                        1 - PPO_EPSILON,
                                                                        1 + PPO_EPSILON) * d_adv),
                                            axis=1))
ovalue_prediction = tf.squeeze(omodel.StateValue, axis=1)
value_prediction = tf.squeeze(cmodel.StateValue, axis=1)
clipped_value_prediction = ovalue_prediction + tf.clip_by_value(value_prediction - ovalue_prediction,
                                                                -PPO_EPSILON,
                                                                PPO_EPSILON)
value_loss = 0.5 * tf.reduce_mean(tf.maximum((value_prediction - d_reward) ** 2,
                                             (clipped_value_prediction - d_reward) ** 2))
entropy_loss = 0.01 * tf.reduce_mean(cmodel.Policies.entropy())
loss = policy_loss + value_loss + entropy_loss

lr = tf.train.polynomial_decay(INITIAL_LR, global_step, MAX_EPOCHS * NUM_MINIBATCH, FINAL_LR)
optimizer = tf.train.RMSPropOptimizer(lr)
grads = tf.gradients(loss, cmodel.variables)
clipped_grads, global_norm = tf.clip_by_global_norm([tf.identity(grad) for grad in grads], 50.0)
grads_n_vars = zip(clipped_grads, cmodel.variables)
train_op = optimizer.apply_gradients(grads_n_vars, global_step=global_step)

tf.summary.scalar("Advantages", tf.reduce_mean(d_sadv))
tf.summary.scalar("GNorm", global_norm)
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
with tf.name_scope("Probability"):
    tf.summary.scalar("Ratio", tf.reduce_max(prob_ratio))

tf.summary.histogram("Cash", d_cash)

summary_op = tf.summary.merge_all()
sync_var_op = model.Model.create_transfer_op(cmodel, omodel)

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(SUMMARY_DIR)

servers = [server.Server(NUM_CLIENTS[idx], PORT + idx,
                         core.PowerTACRolloutHook(model.Model, NUM_CLIENTS[idx], PT_PORT + idx,
                                                  CPU_SEMAPHORE, ROLLOUT_STEPS, ROLLOUT_QUEUE, name="Game%02d" % idx))
           for idx in range(len(NUM_CLIENTS))]

sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(sync_var_op)

server_threads = [threading.Thread(target=server.serve, kwargs={"session": sess}) for server in servers]
utility.apply(lambda thread: thread.start(), server_threads)

state_buffer = numpy.zeros(shape=(BUFFER_SIZE, 2, 1, model.Model.HIDDEN_STATE_COUNT),
                           dtype=numpy.float32)
observation_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS, model.Model.FEATURE_COUNT + 1),
                                 dtype=numpy.float32)
action_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS, model.Model.ACTION_COUNT),
                            dtype=numpy.float32)
reward_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS), dtype=numpy.float32)
value_buffer = numpy.zeros(shape=(BUFFER_SIZE, ROLLOUT_STEPS), dtype=numpy.float32)
nv_buffer = numpy.zeros(shape=(BUFFER_SIZE, 1), dtype=numpy.float32)
rollout_idx = 0
step_count = 0

for rollout in iter(ROLLOUT_QUEUE.get, None):
    state_buffer[rollout_idx: rollout_idx + 1, :, :, :] = numpy.transpose(rollout[0], (2, 0, 1, 3))
    observation_buffer[rollout_idx: rollout_idx + 1, :, :] = numpy.transpose(rollout[1], (1, 0, 2))
    action_buffer[rollout_idx: rollout_idx + 1, :, :] = numpy.transpose(rollout[2], (1, 0, 2))
    reward_buffer[rollout_idx: rollout_idx + 1, :] = numpy.transpose(rollout[3], (1, 0))
    value_buffer[rollout_idx: rollout_idx + 1, :] = numpy.transpose(rollout[4], (1, 0))
    nv_buffer[rollout_idx: rollout_idx + 1, :] = numpy.transpose(rollout[5], (1, 0))

    rollout_idx = (rollout_idx + 1) % BUFFER_SIZE
    if rollout_idx == 0:
        nvalues = numpy.concatenate([value_buffer[:, 1:], nv_buffer], axis=1)
        delta = reward_buffer + gamma(step_count) * nvalues - value_buffer
        delta = numpy.transpose(numpy.transpose(delta)[::-1])
        delta = signal.lfilter([1], [1, -gamma(step_count) * LAMBDA], x=delta)

        if numpy.any(numpy.isnan(observation_buffer)):
            print(numpy.where(numpy.isnan(observation_buffer)))

        advantages = numpy.transpose(numpy.transpose(delta)[::-1])
        rewards = delta + value_buffer

        sess.run([d_iterator.initializer, d_summaryiterator.initializer],
                 feed_dict={state_placeholder: state_buffer,
                            obs_placeholder: observation_buffer,
                            action_placeholder: action_buffer,
                            reward_placeholder: rewards,
                            advantage_placeholder: advantages})
        summary_value = None
        data_present = True
        while data_present:
            try:
                _, summary_value, step_count = sess.run([train_op, summary_op, global_step])
            except tf.errors.OutOfRangeError:
                sess.run(sync_var_op)
                saver.save(sess, CHECKPOINT_PREFIX, global_step=step_count)
                summary_writer.add_summary(summary_value, global_step=step_count)

                update_idx = step_count // (NUM_EPOCHS * NUM_MINIBATCH)
                print("Update %d completed, synchronizing variables" % update_idx)

                utility.apply(lambda hook: hook.update(), map(lambda server: server.Hook, servers))
                data_present = False

        if step_count > MAX_EPOCHS * NUM_MINIBATCH:
            print("Completed training process, terminating session")
            utility.apply(lambda server: server.stop(), servers)
            break


utility.apply(lambda thread: thread.join(), server_threads)
