import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.updates import *
import gym
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import img_as_float, img_as_ubyte
import cPickle as pickle
import itertools
from collections import deque
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import architectures
import time

if __name__ == '__main__':
    
    def exp1_1000k_disk_fix_stale_resumeto500_cuda4():
        # reward sum plateaued to ~-15, so maybe we need a bigger experience replay buffer
        # do: resume from previous experiment both the weights and mem buffer, but just
        # increase the capacity of the memory buffer to 350k
        # NOTE: i accidentally overwrote the weights, so we have to start the weights
        # from scratch but start from the old replay buffer
        lasagne.random.set_rng( np.random.RandomState(0) )
        np.random.seed(0)
        import os
        env = gym.make('Pong-v0')
        env.frameskip = 4
        name = "exp1_1000k_disk_fix_stale_membuffer_resumeto500_cuda4"
        manager = MemoryExperienceManager(filename="weights/exp1_1000k_disk_fix_stale_membuffer.buf", maxlen=500000)
        qq = DeepQ(env,
                   experience_manager=manager,
                   net_fn=dqn_paper_net_fp, net_fn_args={},
                   optimiser=rmsprop, optimiser_args={"learning_rate":0.0002, "rho":0.99},
                   img_preprocessor=preprocessor_pong, debug=True)
        qq.train(update_q=True, render=True if os.environ["USER"] == "cjb60" else False, min_exploration=-1,
                 max_frames=10000000, save_outfile_to="results/%s" % name, save_weights_to="weights/%s.pkl" % name)
