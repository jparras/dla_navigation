import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import platform
from scipy.integrate import simps
from tikzplotlib import save
from dgm_exp import clip, perturbation_np


def obtain_initial_state(obs_low, obs_high):
    return clip(np.random.rand(obs_low.shape[0]) * (obs_high - obs_low) + obs_low, obs_low, obs_high)


if __name__ == '__main__':

    train_DGM = not True
    test_DGM = not True
    plot = True
    exp = 'dgm_exp'  # Experiment name
    test_seeds = 100  # Number of seeds to test
    params_vector = [np.array([0.5, 5, 1]), np.array([0.5, 5, 1]), np.array([0.5, np.pi / 4, 1])]
    pert_mode_vector = ['swirl', 'current_h', 'const']
    n_episodes_test = 100

    # DGM values
    max_pos = 10
    max_vel = 2
    state_low = np.array([-max_pos, -max_pos, -max_vel, -max_vel])
    state_high = np.array([max_pos, max_pos, max_vel, max_vel])

    # Generate base dir for the experiments
    base_dir = os.path.join(os.getcwd(), exp)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # TRAIN DGM (use a single seed, use GPU as well for speed)

    initial_states_file = os.path.join(base_dir, 'initial_states.pickle')
    if os.path.exists(initial_states_file):
        with open(initial_states_file, 'rb') as handle:
            initial_states = pickle.load(handle)['initial_states']
    else:
        initial_states = [obtain_initial_state(state_low, state_high) for _ in range(n_episodes_test)]
        for i in range(n_episodes_test):  # Take initial states only smaller than 8 as initial position
            while np.any(np.abs(initial_states[i][0, 0:2]) > np.array([8, 8])):
                initial_states[i] = obtain_initial_state(state_low, state_high)
        with open(initial_states_file, 'wb') as handle:
            pickle.dump({'initial_states': initial_states}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for i in range(len(pert_mode_vector)):
        if train_DGM or test_DGM:
            print('Obtaining DGM for perturbation ', pert_mode_vector[i])
            if not os.path.exists(os.path.join(base_dir, 'dgm_' + pert_mode_vector[i])):
                os.makedirs(os.path.join(base_dir, 'dgm_' + pert_mode_vector[i]))

            data = {'train': train_DGM,
                    'test': test_DGM,
                    'pert_mode': pert_mode_vector[i],
                    'pert_params': params_vector[i],
                    'dgm_name': os.path.join(base_dir, 'dgm_' + pert_mode_vector[i], 'dgm'),
                    'initial_states': initial_states,
                    'max_pos': max_pos,
                    'max_vel': max_vel,
                    'state_low': state_low,
                    'state_high': state_high
                    }

            # Generate hash for interchange file
            hash = str(random.getrandbits(128))
            data_output = 'data' + hash + '.pickle'
            # Write data output
            with open(data_output, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if platform.system() == 'Windows':
                print("Running on Windows")
                _ = os.system('python dgm_exp.py ' + hash)  # Windows order
            else:
                print("Running on Linux")
                _ = os.system('python3 dgm_exp.py ' + hash)  # Linux order

            # Delete ancillary file
            os.remove(data_output)


    if plot:

        # DGM plot
        eps_to_plot = 10
        nps = 20  # Points to plot perturbation

        st = np.squeeze(np.array(initial_states))
        plt.plot(st[:, 0], st[:, 1], 'ob')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Initial states (position)')
        plt.savefig('initial_states.png')
        save('initial_states.tex')
        plt.show()

        for pmi in range(len(pert_mode_vector)):
            dgm_name = os.path.join(base_dir, 'dgm_' + pert_mode_vector[pmi], 'dgm')
            with open(dgm_name + '_test.pickle', 'rb') as handle:
                data = pickle.load(handle)
            print(pert_mode_vector[pmi])

            ## DGM plot
            with open(dgm_name + '_training.pickle', 'rb') as handle:
                trd = pickle.load(handle)
            for key in trd.keys():
                mean = np.array([np.mean(a) for a in trd[key]])
                plt.semilogy(mean, label=str(key), alpha=0.5)
            plt.xlabel('Training epoch')
            plt.ylabel('Losses')
            plt.legend(loc='best')
            plt.title('Training losses ' + pert_mode_vector[pmi])
            plt.savefig('training_' + pert_mode_vector[pmi] + '.png')
            save('training_' + pert_mode_vector[pmi] + '.tex')
            plt.show()

            for problem in ['orig', 'surr']:
                # Plot the trajectories
                for i in range(eps_to_plot):
                    states = np.squeeze(np.array(data[problem + '_dgm'][i]['states']))
                    plt.plot(states[:, 0], states[:, 1], 'r')  # DGM is red
                    states = np.squeeze(np.array(data[problem + '_bas'][i]['states']))
                    plt.plot(states[:, 0], states[:, 1], 'b')  # baseline is blue
                x = y = np.linspace(state_low[0], state_high[0], nps).reshape([nps, 1])
                X, Y = np.meshgrid(x, y)
                px, py = perturbation_np(pert_mode_vector[pmi], np.ravel(X), np.ravel(Y), params_vector[pmi])
                plt.quiver(np.ravel(X), np.ravel(Y), px.reshape(X.shape) + np.finfo(float).eps, py.reshape(Y.shape) + np.finfo(float).eps)
                plt.plot(0, 0, 'kx')  # Add the origin
                plt.title(problem + ': Position ' + pert_mode_vector[pmi])
                plt.xlabel('x')
                plt.ylabel('y')
                plt.savefig('position_' + pert_mode_vector[pmi] + '_' + problem + '.png')
                save('position_' + pert_mode_vector[pmi] + '_' + problem + '.tex')
                plt.show()


                rc_dgm = []
                fc_dgm = []
                rc_bas = []
                fc_bas = []
                for d in data[problem + '_dgm']:
                    rc_dgm.append(simps(d['running_cost'], d['t'][0: len(d['running_cost'])]))  # Numerically integrate the cost!!
                    fc_dgm.append(d['final_cost'][0])
                for d in data[problem + '_bas']:
                    rc_bas.append(simps(d['running_cost'], d['t'][0: len(d['running_cost'])]))  # Numerically integrate the cost!!
                    fc_bas.append(d['final_cost'][0])
                rc_dgm = np.squeeze(np.array(rc_dgm))
                fc_dgm = np.squeeze(np.array(fc_dgm))
                rc_bas = np.squeeze(np.array(rc_bas))
                fc_bas = np.squeeze(np.array(fc_bas))
                cost_diff = rc_dgm + fc_dgm - rc_bas - fc_bas

                plt.hist(cost_diff, bins=50)
                plt.title('Running cost diff ' + pert_mode_vector[pmi] + ' ' + problem)
                plt.xlabel('seed')
                plt.ylabel('rc')
                save('hist_' + pert_mode_vector[pmi] + '_' + problem + '.tex')
                plt.savefig('hist_' + pert_mode_vector[pmi] + '_' + problem + '.png')
                plt.show()

                print('Total costs for ', pert_mode_vector[pmi], ' ', problem, ' ', np.mean(rc_dgm + fc_dgm), ' (DGM), ', np.mean(rc_bas + fc_bas), ' (bas)')
                print('Cost diff for ', pert_mode_vector[pmi], ' ', problem)
                print('Mean = ', np.mean(cost_diff))
                print('Mean relative improvement ', np.mean(cost_diff / (rc_bas + fc_bas)))
                print('Improvement proportion ', np.sum(cost_diff <= 0) / len(cost_diff))

                # Plot outliers!!
                outliers = list(np.where(cost_diff > 0)[0])
                for i in outliers:
                    states = np.squeeze(np.array(data[problem + '_dgm'][i]['states'], dtype=float))
                    acs = np.squeeze(np.array(data[problem + '_dgm'][i]['actions'], dtype=float))
                    plt.quiver(states[0:-1:10, 0], states[0:-1:10, 1], acs[0::10, 0], acs[0::10, 1], color='g')
                    plt.plot(states[:, 0], states[:, 1], 'r')  # DGM is red
                    states = np.squeeze(np.array(data[problem + '_bas'][i]['states'], dtype=float))
                    acs = np.squeeze(np.array(data[problem + '_bas'][i]['actions'], dtype=float))
                    plt.plot(states[:, 0], states[:, 1], 'b')  # baseline is blue
                    plt.quiver(states[0:-1:10, 0], states[0:-1:10, 1], acs[0::10, 0], acs[0::10, 1], color='orange')
                plt.plot(0, 0, 'kx')  # Add the origin
                plt.title('Outliers ' + problem + ': Position ' + pert_mode_vector[pmi])
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()

                for i in outliers[0:5]:
                    rcd = np.squeeze(np.array(data[problem + '_dgm'][i]['running_cost'], dtype=float))
                    rcb = np.squeeze(np.array(data[problem + '_bas'][i]['running_cost'], dtype=float))
                    plt.plot(rcd, 'r')
                    plt.plot(rcb, 'b')
                    plt.title('Outliers sample ' + problem + ': Cost evolution ' + pert_mode_vector[pmi])
                    plt.xlabel('Time')
                    plt.ylabel('Running Cost')
                    plt.show()
