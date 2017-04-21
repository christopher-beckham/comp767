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

def dqn_paper_net_fp(env, args={}):
    """
    q: the branch of the Q-network, fp: the branch of the future predictor
    """
    def batch_norm_or_not(layer, bn):
        if bn:
            return batch_norm(layer)
        else:
            return layer
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

############ OLD ##############

def dqn_paper_net_spt(env, args={}):
    """
    """
    nonlinearity = rectify if "nonlinearity" not in args else args["nonlinearity"]
    bn = True if "batch_norm" in args else False
    #height, width, nchannels = env.observation_space.shape
    outs = {}
    height, width = 80, 80
    nchannels = 4 # we convert to black and white and use 4 prev frames
    l_in = InputLayer((None, nchannels, height, width))
    l_loc1 = MaxPool2DLayer(l_in, pool_size=2)
    l_loc2 = Conv2DLayer(l_loc1, num_filters=16, filter_size=5)
    l_loc3 = MaxPool2DLayer(l_loc2, pool_size=2)
    l_loc4 = Conv2DLayer(l_loc3, num_filters=16, filter_size=5)
    l_loc5 = MaxPool2DLayer(l_loc4, pool_size=2)
    l_loc6 = Conv2DLayer(l_loc5, num_filters=16, filter_size=5)
    # using identity init for spatial transformer from
    # https://github.com/Lasagne/Recipes/blob/master/examples/spatial_transformer_network.ipynb
    b = np.zeros((2, 3), dtype=theano.config.floatX)
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()
    l_loc_dense = DenseLayer(l_loc6, num_units=6, W=lasagne.init.Constant(0.0), b=b, nonlinearity=identity)
    l_in_transformed = TransformerLayer(l_in, l_loc_dense, downsample_factor=1.0)
    layer = batch_norm_or_not(Conv2DLayer(l_in_transformed, filter_size=8, num_filters=16, stride=4, nonlinearity=nonlinearity), bn)
    layer = batch_norm_or_not(Conv2DLayer(layer, filter_size=4, num_filters=32, stride=2, nonlinearity=nonlinearity), bn)
    # Q branch
    q = DenseLayer(layer, num_units=256, nonlinearity=nonlinearity)  # no bn for a reason
    q = DenseLayer(q, num_units=env.action_space.n, nonlinearity=linear)
    # future prediction
    fp = batch_norm_or_not(Deconv2DLayer(layer, num_filters=16, filter_size=8, stride=2, crop=1), bn)
    fp = batch_norm_or_not(Deconv2DLayer(fp, num_filters=4, filter_size=4, stride=4, nonlinearity=sigmoid), bn)
    return {
        "q": q,
        "fp": fp
    }


def dqn_paper_net_fp_beefier(env, args={}):
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
    dense = DenseLayer(layer, num_units=256, nonlinearity=nonlinearity)  # no bn for a reason
    # q branch
    q = DenseLayer(dense, num_units=env.action_space.n, nonlinearity=linear)
    # future prediction
    fp = ReshapeLayer(DenseLayer(dense, num_units=256), (-1, 4, 8, 8))
    fp = batch_norm_or_not(Deconv2DLayer(fp, num_filters=16, filter_size=8, stride=2, crop=1), bn)
    fp = batch_norm_or_not(Deconv2DLayer(fp, num_filters=4, filter_size=4, stride=4, nonlinearity=sigmoid), bn)
    return {
        "q": q,
        "fp": fp
    }

def dqn_paper_net_fp_inv(env, args={}):
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
    q_final = DenseLayer(q, num_units=env.action_space.n, nonlinearity=linear)
    # future prediction
    fp = inverse_layers(q)
    return {
        "q": q_final,
        "fp": fp
    }

if __name__ == '__main__':
    import gym
    env = gym.make('Pong-v0')
    dd = dqn_paper_net_fp_spt(env)
    print "q"
    l_out = dd['q']
    for layer in get_all_layers(l_out):
        print layer, layer.output_shape
    print count_params(l_out)
