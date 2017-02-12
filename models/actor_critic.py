import logging
import tensorflow as tf
import tft
import toolz
import utils


logging.getLogger('tf').setLevel(logging.ERROR)


def gen_graph_ops(hyper_params):
    hp = hyper_params
    env = utils.EnvWrapper(hp['env'])
    num_actions = env.action_shape[0]

    # TODO fix the state sizes
    state = tf.placeholder(
        tf.float32,
        shape=(None, 4),
        name='state',
    )

    reward = tf.placeholder(
        tf.float32,
        shape=(None, ),
        name='reward',
    )

    action = tf.placeholder(
        tf.int32,
        shape=(None,),
        name='action',
    )

    # actor
    node = tft.nn.affine(state, "output", num_actions)
    action_dist = tf.nn.softmax(node, name="action_dist")
    action_taken = tf.argmax(action_dist, axis=1, name="action_pred")
    action_log_dist = tf.log(action_dist + 1e-6, name="log_action_dist")

    action_mask = tft.utils.boolean_one_hot(action, num_actions)
    masked_log_dist = tf.boolean_mask(action_log_dist, action_mask)
    action_entropy = tf.reduce_sum(action_dist * action_log_dist,
                                   name='entropy_cost')

    # compute value(state)
    value = tft.nn.affine(state, "value_affine", num_units=1)
    value = tf.squeeze(value, name='value')

    # compute advantage
    advantage = tf.sub(reward, tf.stop_gradient(value), name='advantage')
    dot_product = tf.reduce_sum(tf.squeeze(advantage) * masked_log_dist)
    policy_loss = tf.neg(dot_product, name='policy_cost')

    # compute advantage(action, state)
    value_loss = tf.nn.l2_loss(value - reward, name='value_cost')

    # compute overall cost
    beta = tft.nn.hyperparameter(0.01, 'beta')
    cost = tf.add_n([0.5 * value_loss, policy_loss, beta * action_entropy])

    # compute & apply gradient ops
    learning_rate = tft.nn.hyperparameter(0.001, 'learning_rate')
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          decay=0.99,
                                          epsilon=0.1)
    grads_and_vars = optimizer.compute_gradients(cost, params)
    apply_grad = optimizer.apply_gradients(grads_and_vars, name='apply_grads')

    # initializer
    init = tf.global_variables_initializer()

    nodes = [state,
             action,
             action_taken,
             reward,
             value,
             cost,
             apply_grad,
             learning_rate,
             beta,
             ]

    return_map = {op.name.replace(':0', ''): op
                  for op in nodes}
    return_map['grads'] = list(toolz.pluck(0, grads_and_vars))
    return_map['init'] = init
    return return_map
