from models.actor_critic import gen_graph_ops
import numpy as np
import ray
import tensorflow as tf
import toolz
import utils

hyper_params = dict(
    buffer_size=10**5,
    entropy=0.01,
    env='CartPole-v0',
    gamma=0.95,
    hidden_size=64,
    learning_rate=0.001,
    num_iterations=500,
    num_steps=5,
    num_workers=2,
)


def model_init():
    with tf.Graph().as_default():
        name_to_ops = gen_graph_ops(hyper_params)
        sess = tf.Session()
        variables = ray.experimental.TensorFlowVariables(name_to_ops['AddN'],
                                                         sess)
    return {'sess': sess,
            'variables': variables,
            'name_to_ops': name_to_ops}


init_dict = dict(
    buffer=lambda: utils.Buffer(utils.EnvWrapper(hyper_params['env']),
                                hyper_params['buffer_size']),
    env=lambda: utils.EnvWrapper(hyper_params['env']),
    model=model_init
)

ray.init(num_workers=hyper_params['num_workers'])
ray.env.env = ray.EnvironmentVariable(init_dict['env'], toolz.identity)
ray.env.buffer = ray.EnvironmentVariable(init_dict['buffer'], toolz.identity)
ray.env.model_dict = ray.EnvironmentVariable(init_dict['model'],
                                             toolz.identity)


@ray.remote
def step(params):
    env = ray.env.env
    sess = ray.env.model_dict['sess']
    name_to_ops = ray.env.model_dict['name_to_ops']
    variables = ray.env.model_dict['variables']

    def step():
        transition_maps = []
        variables.set_weights(params['weights'])
        for _ in range(hyper_params['num_steps']):
            action = sess.run(name_to_ops['action_pred'],
                              feed_dict={name_to_ops['state']: env.current_state[np.newaxis]})
            transition_map = env.step(action[0])
            transition_maps.append(transition_map)
            if transition_map['is_done']:
                break
        return transition_maps

    def trajectory(batch):
        last_transition = batch[-1]
        is_not_terminal = (1 - last_transition['is_done'])

        target = sess.run(name_to_ops['value'],
                          feed_dict={name_to_ops['state']: last_transition['new_state'][np.newaxis]})
        R = is_not_terminal * target
        batch = utils.chunk_maps(batch)

        clipped = np.clip(batch['reward'], -1, 1)
        rollout = np.append(clipped, R)

        discounted_reward = utils.discount(rollout,
                                           hyper_params['gamma'])
        batch['target'] = discounted_reward[:-1]
        return batch

    batch = step()
    batch = trajectory(batch)
    grads = sess.run(name_to_ops['grads'],
                     {name_to_ops['action']: batch['action'],
                      name_to_ops['state']: batch['initial_state'],
                      name_to_ops['reward']: batch['target']})
    return grads


def train():
    env = ray.env.env
    sess = ray.env.model_dict['sess']
    name_to_ops = ray.env.model_dict['name_to_ops']
    variables = ray.env.model_dict['variables']

    sess.run(name_to_ops['init'])
    params = dict(weights=variables.get_weights())

    procs = []
    for _ in range(hyper_params['num_iterations']):
        with utils.timer('full_loop'):
            jobs = hyper_params['num_workers'] - len(procs)
            procs.extend([step.remote(ray.put(params)) for i in range(jobs)])
            result, procs = ray.wait(procs)
            grads = ray.get(result)[0]
            feed_dict = {name_to_ops['grads'][i]: grads[i]
                         for i in range(len(grads))}
            sess.run(name_to_ops['apply_grads'], feed_dict)
            params['weights'] = variables.get_weights()


if __name__ == '__main__':
    train()
