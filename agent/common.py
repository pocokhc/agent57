import tensorflow as tf
from keras import backend as K
import numpy as np

import enum
import math
import os
import random



def rescaling(x, epsilon=0.001):
    if x == 0:
        return 0
    n = math.sqrt(abs(x)+1) - 1
    return np.sign(x)*n + epsilon*x

def rescaling_inverse(x, epsilon=0.001):
    if x == 0:
        return 0
    n = math.sqrt(1 + 4*epsilon*(abs(x)+1+epsilon)) - 1
    return np.sign(x)*( n/(2*epsilon) - 1)

def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-a * x))

# copy from https://qiita.com/okotaku/items/8d682a11d8f2370684c9
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #session_conf = tf.compat.v1.ConfigProto(
    #    intra_op_parallelism_threads=1,
    #    inter_op_parallelism_threads=1
    #)
    #sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    #tf.compat.v1.keras.backend.set_session(sess)


def create_beta_list(policy_num, max_beta=0.3):
    assert policy_num > 0
    beta_list = []
    for i in range(policy_num):
        if i == 0:
            b = 0
        elif i == policy_num-1:
            b = max_beta
        else:
            b = 10 * (2*i-(policy_num-2)) / (policy_num-2)
            b = max_beta * sigmoid(b)
        beta_list.append(b)
    return beta_list


def create_gamma_list_ngu(policy_num, gamma_min=0.99, gamma_max=0.997):
    assert policy_num > 0
    if policy_num == 1:
        return [gamma_min]
    if policy_num == 2:
        return [gamma_min, gamma_max]
    gamma_list = []
    for i in range(policy_num):
        g = (policy_num - 1 - i)*np.log(1 - gamma_max) + i*np.log(1 - gamma_min)
        g /= policy_num - 1
        g = 1 - np.exp(g)
        gamma_list.append(g)
    return gamma_list


def create_gamma_list_agent57(policy_num, gamma0=0.9999, gamma1=0.997, gamma2=0.99):
    assert policy_num > 0
    gamma_list = []
    for i in range(policy_num):
        if i == 0:
            g = gamma0
        elif 1 <= i and i <= 6:
            g = 10*((2*i-6)/6)
            g = gamma1 + (gamma0 - gamma1)*sigmoid(g)
        elif i == 7:
            g = gamma1
        else:
            g = (policy_num-9)*np.log(1-gamma1) + (i-8)*np.log(1-gamma2)
            g /= policy_num-9
            g = 1-np.exp(g)
        gamma_list.append(g)
    return gamma_list
