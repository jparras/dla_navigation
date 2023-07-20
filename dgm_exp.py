import DGM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
from tikzplotlib import save
import os
import sys


def clip(x, low, high):  # Clip the x vector in each of its dimensions!
    for i in range(x.size):  # Clip state components!
        x[i] = np.clip(x[i], low[i], high[i])
    return x


def transition_surr(state, action, pert_mode, pert_params, dt, kf):
    state = np.squeeze(state)
    action = np.squeeze(action)
    px, py = perturbation_np(pert_mode, state[0], state[1], pert_params=pert_params)
    next_state = state + dt * np.array([state[2], state[3],
                                        action[0] + px - kf * state[2],
                                        action[1] + py - kf * state[3]])
    rc = running_cost(next_state)
    fc = final_cost_np(next_state)

    return next_state.reshape(1, 4), rc, fc


def transition_orig(state, action, pert_mode, pert_params, dt, kf):
    state = np.squeeze(state)
    action = np.squeeze(action)
    px, py = perturbation_np(pert_mode, state[0], state[1], pert_params=pert_params)
    next_state = state + dt * np.array([state[2], state[3],
                                        action[0] + px - kf * state[2],
                                        action[1] + py - kf * state[3]])
    rc = 0
    fc = None

    return next_state.reshape(1, 4), rc, fc


def perturbation_swirl(x, y, params):
    px = (params[0] * (y - params[2]) + np.finfo(float).eps) / tf.sqrt(
        tf.square(x - params[1]) + tf.square(y - params[2]) + np.finfo(float).eps)
    py = - (params[0] * (x - params[1]) + np.finfo(float).eps) / tf.sqrt(
        tf.square(x - params[1]) + tf.square(y - params[2]) + np.finfo(float).eps)
    return px, py


def perturbation_current_h(x, y, params):
    px = params[0] * tf.exp(-tf.square(y - params[1]) / (params[2] ** 2))
    py = tf.zeros_like(y)
    return px, py


def perturbation_const(x, y, params):
    px = params[0] * tf.cast(tf.math.cos(params[1]), tf.float32) * tf.ones_like(x)
    py = params[0] * tf.cast(tf.math.sin(params[1]), tf.float32) * tf.ones_like(y)
    return px, py


def perturbation(pert_mode, x, y, pert_params):
    if pert_mode == 'swirl':
        return perturbation_swirl(x, y, params=pert_params)
    elif pert_mode == 'current_h':
        return perturbation_current_h(x, y, params=pert_params)
    elif pert_mode == 'const':
        return perturbation_const(x, y, params=pert_params)
    elif pert_mode is None:
        return (tf.zeros_like(x), tf.zeros_like(y))
    else:
        raise RuntimeError('Perturbation mode not recognized')


def perturbation_swirl_np(x, y, params):
    px = (params[0] * (y - params[2]) + np.finfo(float).eps) / np.sqrt(
        np.square(x - params[1]) + np.square(y - params[2]) + np.finfo(float).eps)
    py = - (params[0] * (x - params[1]) + np.finfo(float).eps) / np.sqrt(
        np.square(x - params[1]) + np.square(y - params[2]) + np.finfo(float).eps)
    return px, py


def perturbation_current_h_np(x, y, params):
    px = params[0] * np.exp(-np.square(y - params[1]) / (params[2] ** 2))
    py = np.zeros_like(y)
    return px, py


def perturbation_const_np(x, y, params):
    px = params[0] * np.cos(params[1]) * np.ones_like(x)
    py = params[0] * np.sin(params[1]) * np.ones_like(y)
    return px, py


def perturbation_np(pert_mode, x, y, pert_params):
    if pert_mode == 'swirl':
        return perturbation_swirl_np(x, y, params=pert_params)
    elif pert_mode == 'current_h':
        return perturbation_current_h_np(x, y, params=pert_params)
    elif pert_mode == 'const':
        return perturbation_const_np(x, y, params=pert_params)
    elif pert_mode is None:
        return (np.zeros_like(x), np.zeros_like(y))
    else:
        raise RuntimeError('Perturbation mode not recognized')


def sampler(nSim_interior, nSim_terminal, T, t_low, X_high, X_low, state_dim):
    # Sampler #1: domain interior
    t_interior = np.random.uniform(low=t_low - 0.1 * (T - t_low), high=T, size=[nSim_interior, 1])
    X_interior = np.random.uniform(low=X_low - 0.5 * (X_high - X_low), high=X_high + 0.5 * (X_high - X_low),
                                   size=[nSim_interior, state_dim])

    # Sampler #3: initial/terminal condition
    t_terminal = T * np.ones((nSim_terminal, 1))
    X_terminal = np.random.uniform(low=X_low - 0.5 * (X_high - X_low), high=X_high + 0.5 * (X_high - X_low),
                                   size=[nSim_terminal, state_dim])

    return t_interior, X_interior, t_terminal, X_terminal


d_th = 0.5  # Distance indicator
a_cost = -1
b_cost = 1
k_cost = 1


def running_cost(x):  # Implement a differentiable sign function!!
    if x.ndim == 1:
        d = np.sqrt(np.sum(np.square(x[0:2])))
        return np.tanh(k_cost * (d - d_th)) * (b_cost-a_cost)/2 + (b_cost+a_cost)/2

    else:
        d = np.sqrt(np.sum(np.square(x[:, 0:2]), axis=1))
        return np.tanh(k_cost * (d - d_th)) * (b_cost-a_cost)/2 + (b_cost+a_cost)/2


def final_cost_np(x):
    if x.ndim == 1:
        return 0
    else:
        return np.zeros(x.shape[0])


def loss(model, t_interior, X_interior, t_terminal, X_terminal, control, pert_mode=None, pert_params=None, kf=0):
    # Loss term #1: PDE
    # compute function value and derivatives at current sampled points
    V = model(t_interior, X_interior)
    V_t = tf.gradients(V, t_interior)[0]
    V_x = tf.gradients(V, X_interior)[0]
    px, py = perturbation(pert_mode, X_interior[:, 0], X_interior[:, 1], pert_params=pert_params)
    trans = tf.concat([tf.expand_dims(X_interior[:, 2], -1),
                       tf.expand_dims(X_interior[:, 3], -1),
                       tf.math.cos(control) + tf.expand_dims(px, -1) - kf * tf.expand_dims(X_interior[:, 2], -1),
                       tf.math.sin(control) + tf.expand_dims(py, -1) - kf * tf.expand_dims(X_interior[:, 3], -1)], 1)
    sum = tf.expand_dims(tf.reduce_sum(V_x * trans, axis=1), -1)
    d = tf.expand_dims(tf.math.sqrt(tf.reduce_sum(tf.square(X_interior[:, 0:2]), axis=1)), -1)
    running_cost = tf.math.tanh(k_cost * (d - d_th)) * (b_cost-a_cost)/2 + (b_cost+a_cost)/2
    diff_V = V_t + sum + running_cost  # Hamiltonian

    # compute average L2-norm of differential operator
    L1 = tf.reduce_mean(tf.square(diff_V))

    # Loss term #3: initial/terminal condition
    fitted_terminal = model(t_terminal, X_terminal)

    L3 = tf.reduce_mean(tf.square(fitted_terminal))  # Final cost is 0!
    L_control = tf.reduce_mean(sum + running_cost)

    return L1, L3, L_control, running_cost, fitted_terminal


def compute_fitted_optimal_control(model, state_tnsr, time_tnsr):
    V = model(time_tnsr, state_tnsr)
    V_x = tf.gradients(V, state_tnsr)[0]
    control = tf.math.atan2(-V_x[:, 3], -V_x[:, 2])

    return tf.expand_dims(control, -1)


def save_variables(save_path, variables=None, sess=None):
    import joblib
    sess = sess
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)


def load_variables(load_path, variables=None, sess=None):
    import joblib
    sess = sess
    variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[v.name]))
    sess.run(restores)


class DeepGalerkinMethod(object):
    def __init__(self, T=1, state_dim=1, t_low=1e-10, state_low=-1, state_high=1, num_layers=3, nodes_per_layer=50,
                 pert_mode=None, pert_params=None, saveName=None, kf=0):
        # %% Parameters
        self.T = T  # terminal time
        self.state_dim = state_dim
        # Solution parameters (domain on which to solve PDE)
        self.t_low = t_low
        self.state_low = state_low
        self.state_high = state_high
        self.kf = kf

        # neural network parameters
        self.num_layers = num_layers  # Number of layers used in DGM model
        self.nodes_per_layer = nodes_per_layer  # Output dim of each layer

        # Perturbation parameters
        self.pert_mode = pert_mode
        self.pert_params = pert_params

        # initialize DGM model (last input: space dimension = 1)
        self.model = DGM.DGMNet(self.nodes_per_layer, self.num_layers, self.state_dim, scope='DGM_value_function')

        # tensor placeholders (_tnsr suffix indicates tensors)
        # inputs (time, space domain interior, space domain at initial time)
        self.t_interior_tnsr = tf.placeholder(tf.float32, [None, 1])
        self.state_interior_tnsr = tf.placeholder(tf.float32, [None, self.state_dim])
        self.t_terminal_tnsr = tf.placeholder(tf.float32, [None, 1])
        self.state_terminal_tnsr = tf.placeholder(tf.float32, [None, self.state_dim])

        # optimal control computed numerically from fitted value function
        self.control = compute_fitted_optimal_control(self.model, self.state_interior_tnsr, self.t_interior_tnsr)

        # loss
        self.L1_tnsr, self.L3_tnsr, self.L_control, self.rc, self.fc = loss(self.model, self.t_interior_tnsr,
                                                                            self.state_interior_tnsr,
                                                                            self.t_terminal_tnsr,
                                                                            self.state_terminal_tnsr, self.control,
                                                                            pert_mode=self.pert_mode,
                                                                            pert_params=self.pert_params,
                                                                            kf=self.kf)
        self.loss_tnsr = self.L1_tnsr + self.L3_tnsr

        # value function
        self.V = self.model(self.t_interior_tnsr, self.state_interior_tnsr)
        self.V_t = tf.gradients(self.V, self.t_interior_tnsr)[0]
        self.V_x = tf.gradients(self.V, self.state_interior_tnsr)[0]

        self.value_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DGM_value_function")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss_tnsr, var_list=self.value_vars)

        # initialize variables
        init_op = tf.global_variables_initializer()

        # open session
        self.sess = tf.Session()
        self.sess.run(init_op)
        if saveName is not None:
            load_variables(saveName, variables=self.value_vars, sess=self.sess)
            # saver = tf.train.Saver(var_list=self.value_vars)
            # saver.restore(self.sess, saveName)

    def train(self, sampling_stages=250, steps_per_sample=10, nSim_interior=10000, nSim_terminal=10000, saveName=None):

        # %% Train network
        # initialize loss per training
        loss_list = []
        l1_list = []
        l3_list = []
        print('TRAINING DGM')
        # for each sampling stage
        for i in range(sampling_stages):

            # sample uniformly from the required regions
            t_interior, X_interior, t_terminal, X_terminal = sampler(nSim_interior, nSim_terminal, self.T, self.t_low,
                                                                     self.state_high, self.state_low, self.state_dim)

            # for a given sample, take the required number of SGD steps
            loss_aux = []
            l1_aux = []
            l3_aux = []
            for _ in range(steps_per_sample):
                loss, L1, L3, _ = self.sess.run([self.loss_tnsr, self.L1_tnsr, self.L3_tnsr, self.optimizer],
                                                feed_dict={self.t_interior_tnsr: t_interior,
                                                           self.state_interior_tnsr: X_interior,
                                                           self.t_terminal_tnsr: t_terminal,
                                                           self.state_terminal_tnsr: X_terminal})
                loss_aux.append(loss)
                l1_aux.append(L1)
                l3_aux.append(L3)
            loss_list.append(loss_aux)
            l1_list.append(l1_aux)
            l3_list.append(l3_aux)
            print('DGM: Iteration = ', i, '; Total loss = ', loss, '; Ham loss = ', L1, '; Final value loss = ', L3)

        if saveName is not None:
            # saver = tf.train.Saver(var_list=self.value_vars)
            # saver.save(self.sess, saveName)
            save_variables(saveName, variables=self.value_vars, sess=self.sess)
            with open(saveName + '_training.pickle', 'wb') as handle:
                pickle.dump({'loss': loss_list, 'l1': l1_list, 'l3': l3_list}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_dgm(self, saveName=None, show=False):  # Generates and saves plot info to be used afterwards

        data = {}

        # First, obtain a plot for the value function
        nps = 100
        npt = 5
        x = y = np.linspace(self.state_low[0], self.state_high[0], nps).reshape([nps, 1])
        t = np.linspace(self.t_low, self.T, npt).reshape([npt, 1])
        X, Y = np.meshgrid(x, y)
        states = np.vstack([np.ravel(X), np.ravel(Y), np.zeros_like(np.ravel(X)), np.zeros_like(np.ravel(Y))]).T
        data['v'] = {'x': X, 'y': Y, 't': t, 'v': []}
        for it in range(npt):
            data['v']['v'].append(
                self.sess.run([self.V], feed_dict={self.t_interior_tnsr: t[it] * np.ones((nps * nps, 1)),
                                                   self.state_interior_tnsr: states})[0])
         # Second, obtain a plot for the actions
        nps = 20
        npt = 11
        x = y = np.linspace(self.state_low[0], self.state_high[0], nps).reshape([nps, 1])
        t = np.linspace(self.t_low, self.T, npt).reshape([npt, 1])
        X, Y = np.meshgrid(x, y)
        states = np.vstack([np.ravel(X), np.ravel(Y), np.zeros_like(np.ravel(X)), np.zeros_like(np.ravel(Y))]).T
        data['acs'] = {'x': X, 'y': Y, 't': t, 'acs': []}
        for it in range(npt):
            data['acs']['acs'].append(self.sess.run([[self.control]],
                                                    feed_dict={
                                                        self.t_interior_tnsr: t[it] * np.ones((nps * nps, 1)),
                                                        self.state_interior_tnsr: states})[0][0])

        if saveName is not None:
            with open(saveName + '.pickle', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if show:  # Show data
            for it in range(len(data['v']['t'])):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(data['v']['x'], data['v']['y'], data['v']['v'][it].reshape(data['v']['x'].shape))
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('V')
                plt.savefig(saveName + '_v_' + str(data['v']['t'][it]) + '.png')
                plt.show()

            ys = [i + data['acs']['t'] + (i * data['acs']['t']) ** 2 for i in range(len(data['acs']['t']))]
            cmap = cm.get_cmap('rainbow')
            colors = cmap(np.linspace(0, 1, len(ys)))
            for it in range(len(data['acs']['t'])):
                if it == len(data['acs']['t']) - 1:
                    plt.quiver(np.ravel(data['acs']['x']), np.ravel(data['acs']['y']),
                               np.cos(data['acs']['acs'][it]), np.sin(data['acs']['acs'][it]),
                               label=str(data['acs']['t'][it]), color=colors[it], alpha=0.3)
                else:
                    plt.quiver(np.ravel(data['acs']['x']), np.ravel(data['acs']['y']),
                               np.cos(data['acs']['acs'][it]), np.sin(data['acs']['acs'][it]),
                               label=str(data['acs']['t'][it]), color=colors[it])
            plt.savefig(saveName + '_acs_' + '.png')
            save(saveName + '_acs_' + '.tex')
            plt.show()


if __name__ == '__main__':

    hash = sys.argv[1]
    data_input = 'data' + hash + '.pickle'
    with open(data_input, 'rb') as handle:
       data = pickle.load(handle)


    T = 10
    state_dim = 4
    max_pos = data['max_pos']
    max_vel = data['max_vel']
    state_low = data['state_low']
    state_high = data['state_high']
    kf = 0.5
    time_step = 0.01
    t_low = 1e-10

    pert_mode = data['pert_mode']
    pert_params = data['pert_params']
    dgm_name = data['dgm_name']
    initial_states = data['initial_states']

    if data['train']:

        dgm = DeepGalerkinMethod(T=T, t_low=t_low, state_dim=state_dim, state_low=state_low, state_high=state_high,
                                 saveName=None, num_layers=3, nodes_per_layer=50, pert_mode=pert_mode,
                                 pert_params=pert_params, kf=kf)
        dgm.train(sampling_stages=10000, steps_per_sample=10, nSim_interior=10000, nSim_terminal=10000,
                  saveName=dgm_name)

    else:
        dgm = DeepGalerkinMethod(T=T, t_low=t_low, state_dim=state_dim, state_low=state_low, state_high=state_high,
                                 saveName=dgm_name, num_layers=3, nodes_per_layer=50, pert_mode=pert_mode,
                                 pert_params=pert_params, kf=kf)
    # Plot data

    dgm.plot_dgm(saveName=dgm_name, show=True)

    # Test and save data!!
    n_episodes = len(initial_states)
    out = {}
    tv = np.arange(t_low, T, time_step).astype(list)


    def get_rc(state, problem):
        if problem is 'orig':
            return 0
        else:
            return running_cost(state)[0]


    def get_ac(state, t, policy):
        if policy is 'dgm':
            action, rc = dgm.sess.run([dgm.control, dgm.rc],
                                      feed_dict={dgm.t_interior_tnsr: t * np.ones((1, 1)),
                                                 dgm.state_interior_tnsr: state})
            return np.squeeze(np.array([np.cos(action), np.sin(action)]))
        else:
            ac = np.arctan2(-state[0, 1], -state[0, 0])
            return np.array([np.cos(ac), np.sin(ac)])


    def get_transition(state, action, problem):
        if problem is 'orig':
            next_state, rc, fc = transition_orig(state, action, pert_mode, pert_params, time_step, kf)
        else:
            next_state, rc, fc = transition_surr(state, action, pert_mode, pert_params, time_step, kf)
        return next_state, rc, fc


    for policy in ['dgm', 'bas']:
        for problem in ['orig', 'surr']:
            output_data = []

            for e in range(n_episodes):
                print('Testing for policy ', policy, '; problem ', problem, '; episode ', e, ' of ', n_episodes)
                state = clip(np.squeeze(initial_states[e]), state_low, state_high).reshape(1, state_low.shape[0])
                done = False

                episode_data = {'states': [state],
                                'actions': [],
                                'running_cost': [get_rc(state, problem)],
                                'final_cost': [],
                                't': np.append(tv, T)
                                }
                for t in tv:
                    action = get_ac(state, t, policy)
                    # Perform action
                    next_state, rc, fc = get_transition(state, action, problem)
                    state = next_state

                    # Save data
                    episode_data['states'].append(state)
                    episode_data['actions'].append(action)
                    episode_data['running_cost'].append(rc)

                    # Check for early stopping!
                    if problem is 'orig':
                        if np.sqrt(np.sum(np.square(next_state[0, 0:2]))) < d_th or t is tv[-1]:
                            episode_data['final_cost'].append(t)
                            break

                if problem is 'surr':
                    episode_data['final_cost'].append(fc)  # Append final cost!
                output_data.append(episode_data)

            out[problem + '_' + policy] = output_data

    with open(dgm_name + '_test.pickle', 'wb') as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

