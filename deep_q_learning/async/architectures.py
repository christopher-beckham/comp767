import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.updates import *
import numpy as np

def batch_norm_or_not(layer, bn):
    if bn:
        return batch_norm(layer)
    else:
        return layer

def inverse_layers(l_out):
    l_inv = l_out
    for layer in get_all_layers(l_out)[::-1]:
        if isinstance(layer, InputLayer):
            break
        l_inv = InverseLayer(l_inv, layer)
    return l_inv


def dqn_paper_net(env, args={}):
    """
    """
    nonlinearity = rectify if "nonlinearity" not in args else args["nonlinearity"]
    bn = True if "batch_norm" in args else False
    #height, width, nchannels = env.observation_space.shape
    outs = {}
    height, width = 80, 80
    nchannels = 4 # we convert to black and white and use 4 prev frames
    layer = InputLayer((None, nchannels, height, width))
    layer = batch_norm_or_not(Conv2DLayer(layer, filter_size=8, num_filters=16, stride=4, nonlinearity=nonlinearity), bn)
    layer = batch_norm_or_not(Conv2DLayer(layer, filter_size=4, num_filters=32, stride=2, nonlinearity=nonlinearity), bn)
    # Q branch
    q = DenseLayer(layer, num_units=256, nonlinearity=nonlinearity)  # no bn for a reason
    q = DenseLayer(q, num_units=env.action_space.n, nonlinearity=linear)
    return q

if __name__ == '__main__':
    """
    import gym
    env = gym.make('Pong-v0')
    dd = dqn_paper_net_fp_spt(env)
    print "q"
    l_out = dd['q']
    for layer in get_all_layers(l_out):
        print layer, layer.output_shape
    print count_params(l_out)
    """
    pass
