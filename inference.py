import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle5 as pickle
from AdEx_const import *
import os
from tqdm.notebook import tqdm_notebook
import time


class snn_pc(object):

    def __init__(self, w_mat,
                 num_pc_layers=3,
                 num_pred_neurons=[1296, 1156, 1024], num_gist_neurons=128, num_input_neurons=784,
                 sim_duration=350e-3, sim_dt=1e-4):

        # network architecture
        self.n_pc_layer = num_pc_layers
        self.n_pred = num_pred_neurons
        self.n_gist = num_gist_neurons
        self.n_stim = num_input_neurons

        # self.n_groups = num_pc_layers * 3 + 1
        self.neurons_per_group = [self.n_stim] * 3 + np.repeat([self.n_pred[:-1]], 3).tolist() + [self.n_pred[-1]] + [
            self.n_gist]
        self.n_variable = sum(self.neurons_per_group)

        # initial weight preparation
        self.w = w_mat

        # simulation parameters
        self.T = sim_duration
        self.dt = sim_dt

        print ('SNN-PC initialized')

    def initialize_var(self):

        # internal variables
        self.v = tf.Variable(tf.ones([self.n_variable, 1], dtype=tf.float32) * EL)
        self.c = tf.Variable(tf.zeros([self.n_variable, 1], dtype=tf.float32))
        self.ref = tf.Variable(tf.zeros([self.n_variable, 1], dtype=tf.float32))
        # pre-synaptic variables
        self.x = tf.Variable(tf.zeros([self.n_variable, 1], dtype=tf.float32))
        self.x_tr = tf.Variable(tf.zeros([self.n_variable, 1], dtype=tf.float32))
        # post-synaptic variable
        self.Isyn = tf.Variable(tf.zeros([self.n_variable, 1], dtype=tf.float32))
        self.fired = tf.Variable(tf.zeros([self.n_variable, 1], dtype=tf.bool))

        self.xtr_record = tf.Variable(tf.zeros([self.n_variable, 1], dtype=tf.float32))

        print ('Reset to resting state')

    def __call__(self, I_ext, plot_window=10e-3):

        # simulation parameters
        self._step = 0

        # initialize internal variables
        self.initialize_var()

        # preprocess_input
        ext_input = preprocess_input(test_img=I_ext, new_max=3000e-12, new_min=600e-12)
        # feed external corrent to the first layer
        self.Iext = tf.reshape(tf.constant(ext_input, dtype=tf.float32), shape=(self.n_stim, 1))
        # open up a dictionary to save plots
        self.live_imgs = {'pc1': tf.Variable(tf.zeros(shape=(int(self.T / plot_window) + 1, 3, self.n_stim))),
                          'pc2': tf.Variable(tf.zeros(shape=(int(self.T / plot_window) + 1, 3, self.n_pred[0]))),
                          'pc3': tf.Variable(tf.zeros(shape=(int(self.T / plot_window) + 1, 3, self.n_pred[1])))
                          }
        save_idx = 1

        print ('Simulation in progress. Please wait.')

        for t in tqdm_notebook(range(int(self.T / self.dt))):
            # update internal variables (v, c, x, x_tr)
            self.update_var()

            if (t+1) % int(plot_window / self.dt) == 0:

                self.save_activity(save_idx=save_idx, syn_curr=self.xtr_record/self._step)
                save_idx += 1

            else:
                self.xtr_record.assign_add(self.x_tr)

            self._step += 1
            # time.sleep(0.001)

        print('Simulation finished!')

    def update_var(self):

        # feed synaptic current to higher layers
        self.update_Isyn()

        # current refractory status [0,2] ms
        ref_constraint = tf.cast(tf.greater(self.ref, 0), tf.float32)
        # update v according to ref: if in ref, dv = 0
        self.update_v(ref_constraint)
        self.update_c(ref_constraint)

        # subtract one time step (1) from refractory vector
        self.ref = tf.cast(tf.maximum(tf.subtract(self.ref, 1), 0), tf.float32)

        # update synaptic current
        self.update_x()
        self.update_xtr()

        # update spike monitor (fired: dtype=bool): if fired = True, else = False
        self.fired = tf.cast(tf.greater_equal(self.v, VT), tf.float32)
        # self.fs.assign_add(self.fired)
        # reset variables
        self.v = self.fired * EL + (1 - self.fired) * self.v
        self.c = self.fired * tf.add(self.c, b) + (1 - self.fired) * self.c
        self.x = self.fired * -x_reset + (1 - self.fired) * self.x

        # self.update_xtr()

        # set lower boundary of v (Vrest = -70.6 mV)
        self.v = tf.maximum(EL, self.v)
        self.ref = tf.add(self.ref, self.fired * float(t_ref / self.dt))

    def update_v(self, constraint):
        dv = (self.dt / Cm) * (gL * (EL - self.v) +
                                    gL * DeltaT * tf.exp((self.v - VT) / DeltaT) +
                                    self.Isyn - self.c)
        dv_ref = (1 - constraint) * dv
        self.v = tf.add(self.v, dv_ref)

    def update_c(self, constraint):
        dc = (self.dt / tauw) * (a * (self.v - EL) - self.c)
        dc_ref = (1 - constraint) * dc
        self.c = tf.add(self.c, dc_ref)

    def update_x(self):
        dx = self.dt * (-self.x / tau_rise)
        self.x = tf.add(self.x, dx)

    def update_xtr(self):
        dxtr = self.dt * (-self.x / tau_rise - self.x_tr / tau_s)
        self.x_tr = tf.add(self.x_tr, dxtr)

    def update_Isyn(self):

        # I = ext
        self.Isyn[:self.n_stim].assign(
            self.Iext)
        # gist = W[ig]@ Isyn[I]
        input_gist = tf.transpose(self.w['ig']) @ (self.x_tr[:self.neurons_per_group[0]])
        self.Isyn[-self.n_gist:, :].assign(input_gist)

        for pc_layer_idx in range(self.n_pc_layer):
            self.Isyn_by_layer(pc_layer_idx, wc=tf.cast(tf.greater(self._step - 500, 0), tf.float32))

    def Isyn_by_layer(self, pc_layer_idx, wc):
        # index of current prediction layer
        curr_p_idx = sum(self.neurons_per_group[:pc_layer_idx * 3])
        curr_p_size = self.neurons_per_group[pc_layer_idx * 3]

        # index of next prediction layer
        next_p_idx = sum(self.neurons_per_group[:pc_layer_idx * 3 + 3])
        next_p_size = self.neurons_per_group[pc_layer_idx * 3 + 3]

        # input / prediction error
        bu_sensory = wc * (
            self.x_tr[curr_p_idx: curr_p_idx + curr_p_size, :])
        # prediction
        td_pred = wc * (
            self.w['pc' + str(pc_layer_idx + 1)] @ self.x_tr[next_p_idx:next_p_idx + next_p_size, :])

        # E+ = I - P
        self.Isyn[curr_p_idx + curr_p_size:curr_p_idx + 2 * curr_p_size, :].assign(
                tf.add(bu_sensory, -td_pred) + 600 * pamp)
        # E- = -I + P
        self.Isyn[curr_p_idx + 2 * curr_p_size:next_p_idx, :].assign(
            tf.add(-bu_sensory, td_pred) + 600 * pamp)

        # P = bu_error + td_error + gist
        bu_err_pos = tf.transpose(self.w['pc' + str(pc_layer_idx + 1)]) @ (
            self.x_tr[curr_p_idx + curr_p_size:curr_p_idx + 2 * curr_p_size, :] * wc)
        bu_err_neg = tf.transpose(self.w['pc' + str(pc_layer_idx + 1)]) @ (
                self.x_tr[curr_p_idx + 2 * curr_p_size:next_p_idx, :] * wc)
        gist = tf.transpose(self.w['gp' + str(pc_layer_idx + 1)]) @ self.x_tr[-self.n_gist:, :]

        if pc_layer_idx < self.n_pc_layer - 1:
            td_err_pos = self.x_tr[next_p_idx + next_p_size:next_p_idx + 2 * next_p_size] * wc#self.w_const
            td_err_neg = self.x_tr[next_p_idx + 2 * next_p_size:next_p_idx + 3 * next_p_size] * wc#self.w_const
            self.Isyn[next_p_idx:next_p_idx + next_p_size, :].assign(
                tf.add(
                    tf.add(
                        tf.add(bu_err_pos, -bu_err_neg),
                        tf.add(-td_err_pos, td_err_neg)),
                    gist))
        else:
            self.Isyn[next_p_idx:next_p_idx + next_p_size, :].assign(
                tf.add(
                    tf.add(
                        bu_err_pos, -bu_err_neg),
                    gist))

    def save_activity(self, save_idx, syn_curr):

        # loop over pc layers
        for pc_i in range(self.n_pc_layer):

            # index
            bu_start_idx = sum(self.neurons_per_group[:pc_i * 3])
            bu_end_idx = sum(self.neurons_per_group[:pc_i * 3 + 1])

            td_start_idx = sum(self.neurons_per_group[:(pc_i + 1) * 3])
            td_end_idx = sum(self.neurons_per_group[:(pc_i + 1) * 3 + 1])

            # synaptic current
            bu_input = syn_curr[bu_start_idx:bu_end_idx] / pamp
            td_prediction = (self.w['pc' + str(pc_i + 1)] @ syn_curr[td_start_idx:td_end_idx]) / pamp

            # col 0: BU input, col1: error, col2: TD pred
            self.live_imgs['pc' + str(pc_i + 1)][save_idx, 0].assign(tf.reshape(bu_input, (bu_input.shape[0],)))
            self.live_imgs['pc' + str(pc_i + 1)][save_idx, 1].assign(tf.reshape(bu_input - td_prediction, (bu_input.shape[0],)))
            self.live_imgs['pc' + str(pc_i + 1)][save_idx, 2].assign(tf.reshape(td_prediction, (bu_input.shape[0],)))

def load_and_convert_weights(pickle_path):
    # load learned weights from previous training
    with open(pickle_path, 'rb') as wdict:
        weights_mat = pickle.load(wdict)
    # convert them to tensors
    for key, grp in weights_mat.items():
        weights_mat[key] = tf.convert_to_tensor(grp)
    return weights_mat

def preprocess_input(test_img, new_max, new_min):

    # normalize into a unit vector
    norm_digits = test_img / np.linalg.norm(test_img)
    # scale to [0, 1]
    div_a = norm_digits - np.min(norm_digits)
    div_b = np.max(norm_digits) - np.min(norm_digits)
    scaled = np.divide(div_a, div_b, out=np.zeros_like(div_a), where=div_b != 0)
    # scale to [min, max]
    new_diff = new_max - new_min

    return scaled * new_diff + new_min

def create_noisey_sample(choice_digit, choice_sample, choice_noise):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255

    sample_indices = np.where(y_test == int(choice_digit))[0]
    sample_chosen = sample_indices[choice_sample]

    sample = tf.keras.layers.GaussianNoise(choice_noise)
    noisey = sample(x_test[sample_chosen].astype(np.float32), training=True)

    return noisey