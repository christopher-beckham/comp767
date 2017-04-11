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

    import sys

    def preprocessor_pong(img):
        img = rgb2gray(img) # (210, 160)
        img = resize(img, (img.shape[0]//2, img.shape[1]//2)) # (105, 80)
        img = img[17:97,:] # (80, 80)
        img = img_as_ubyte(img) # to save on memory
        return img

    #optimiser=rmsprop,
    #optimiser_args={"learning_rate":.0002, "rho":0.99},
 
    # these experiments had the uint8 bug
        
    """
    def exp1_1000k_disk():
        lasagne.random.set_rng( np.random.RandomState(0) )
        np.random.seed(0)
        import os
        env = gym.make('Pong-v0')
        env.frameskip = 4
        name = "exp1_1000k_disk"
        manager = DiskExperienceManager(filename="weights/%s" % name, maxlen=1000000)
        qq = DeepQ(env,
                   experience_manager=manager,
                   net_fn=dqn_paper_net_fp, net_fn_args={},
                   optimiser=rmsprop, optimiser_args={"learning_rate":0.0002, "rho":0.99},
                   img_preprocessor=preprocessor_pong, debug=True)
        qq.train(update_q=True, render=True if os.environ["USER"] == "cjb60" else False, min_exploration=-1,
                 max_frames=10000000, save_outfile_to="results/%s" % name, save_weights_to="weights/%s.pkl" % name)


    def exp1_1000k_disk_fix_stale():
        # https://github.com/christopher-beckham/comp767/compare/34bb65cd9ca266ee3af1ee360ecaac67315dafb2...04cb0a857143bd69a703177044f4cd4dd9320da3
        # basically: l_out_stale was equal to l_out when it shouldn't have been...
        # maybe this was the cause of the bug
        # ----
        # we are starting from scratch using a MemoryExperienceManager because the sampling
        # time for DiskExperienceManager increases w.r.t. the size of the buffer
        lasagne.random.set_rng( np.random.RandomState(0) )
        np.random.seed(0)
        import os
        env = gym.make('Pong-v0')
        env.frameskip = 4
        name = "exp1_1000k_disk_fix_stale_membuffer"
        #manager = DiskExperienceManager(filename="weights/exp1_1000k_disk", maxlen=1000000, flag="r")
        manager = MemoryExperienceManager(filename="weights/%s.pkl" % name, maxlen=200000)
        qq = DeepQ(env,
                   experience_manager=manager,
                   net_fn=dqn_paper_net_fp, net_fn_args={},
                   optimiser=rmsprop, optimiser_args={"learning_rate":0.0002, "rho":0.99},
                   img_preprocessor=preprocessor_pong, debug=True)
        qq.train(update_q=True, render=True if os.environ["USER"] == "cjb60" else False, min_exploration=-1,
                 max_frames=10000000, save_outfile_to="results/%s" % name, save_weights_to="weights/%s.pkl" % name)



    def exp1_1000k_disk_fix_stale_resumeto300():
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
        name = "exp1_1000k_disk_fix_stale_membuffer_resumeto300"
        manager = MemoryExperienceManager(filename="weights/exp1_1000k_disk_fix_stale_membuffer.buf", maxlen=350000)
        qq = DeepQ(env,
                   experience_manager=manager,
                   net_fn=dqn_paper_net_fp, net_fn_args={},
                   optimiser=rmsprop, optimiser_args={"learning_rate":0.0002, "rho":0.99},
                   img_preprocessor=preprocessor_pong, debug=True)
        qq.train(update_q=True, render=True if os.environ["USER"] == "cjb60" else False, min_exploration=-1,
                 max_frames=10000000, save_outfile_to="results/%s" % name, save_weights_to="weights/%s.pkl" % name)


    # experiments where we use the 'old weights' as a base (for time sake)
        
    def exp1_1000k_disk_fix_stale_resumeto300_useoldwt():
        # reward sum plateaued to ~-15, so maybe we need a bigger experience replay buffer
        # do: resume from previous experiment both the weights and mem buffer, but just
        # increase the capacity of the memory buffer to 350k
        # NOTE: we use the weights from the best model (a couple weeks ago) that 'magically
        # worked' when the implementation was bugged (it's the best thing we have so far)
        lasagne.random.set_rng( np.random.RandomState(0) )
        np.random.seed(0)
        import os
        env = gym.make('Pong-v0')
        env.frameskip = 4
        name = "exp1_1000k_disk_fix_stale_membuffer_resumeto300_useoldwt"
        manager = MemoryExperienceManager(filename="weights/exp1_1000k_disk_fix_stale_membuffer.buf", maxlen=350000)
        qq = DeepQ(env,
                   experience_manager=manager,
                   net_fn=dqn_paper_net_fp, net_fn_args={},
                   optimiser=rmsprop, optimiser_args={"learning_rate":0.0002, "rho":0.99},
                   img_preprocessor=preprocessor_pong, debug=True)
        qq.load_weights_from("weights/dqn_paper_revamp2_bn_sgd_noclip_2_rmsprop_bigexperience.pkl", legacy=True)
        qq.train(update_q=True, render=True if os.environ["USER"] == "cjb60" else False, min_exploration=-1,
                 eps_max=0.5, eps_min=0.1,
                 max_frames=10000000, save_outfile_to="results/%s" % name, save_weights_to="weights/%s.pkl" % name)
    """

    def exp1_1000k_disk_fix_stale_resumeto300_useoldwt_fixint():
        # start with 200k-sized mem buffer from old experiment, and increase mem buffer capacity to 350k
        # also use pre-trained weights from an old experiment that hit the best reward sum of -13
        lasagne.random.set_rng( np.random.RandomState(0) )
        np.random.seed(0)
        import os
        env = gym.make('Pong-v0')
        env.frameskip = 4
        name = "exp1_1000k_disk_fix_stale_membuffer_resumeto300_useoldwt_fixint"
        manager = MemoryExperienceManager(filename="weights/exp1_1000k_disk_fix_stale_membuffer.buf", maxlen=350000)
        qq = DeepQ(env,
                   experience_manager=manager,
                   net_fn=dqn_paper_net_fp, net_fn_args={},
                   optimiser=rmsprop, optimiser_args={"learning_rate":0.0002, "rho":0.99},
                   img_preprocessor=preprocessor_pong, debug=True)
        qq.load_weights_from("weights/dqn_paper_revamp2_bn_sgd_noclip_2_rmsprop_bigexperience.pkl", legacy=True)
        qq.train(update_q=True, render=True if os.environ["USER"] == "cjb60" else False, min_exploration=-1,
                 eps_max=0.5, eps_min=0.1,
                 max_frames=10000000, save_outfile_to="results/%s" % name, save_weights_to="weights/%s.pkl" % name)


    """
    def dqn_paper_adam_again_noclip_fp():
        #
        #For 'FP': same as above but with future prediction with fp_lambda=1.
        #(which seemed to be too much)
        #NOTE: since this was run before 03/04, it has the 'off-by-one'
        #bug with the tuples. So it might be a good idea to re-run this,
        #if time-permitting.
        #
        lasagne.random.set_rng( np.random.RandomState(0) )
        np.random.seed(0)
        import os
        env = gym.make('Pong-v0')
        env.frameskip = 4
        name = "dqn_paper_revamp2_bn_sgd_noclip_2_rmsprop_bigexperience_fp_beefier"
        qq = DeepQ(env, net_fn=architectures.dqn_paper_net_fp_beefier, net_fn_args={}, optimiser=rmsprop, optimiser_args={"learning_rate":0.0002, "rho":0.99},
                   img_preprocessor=preprocessor_pong, lambda_fp=1.0, debug=True, experience_maxlen=500000)
        qq.train(update_q=True, render=True if os.environ["USER"] == "cjb60" else False, min_exploration=-1,
                 max_frames=10000000, save_outfile_to="results/%s" % name, save_weights_to="weights/%s.pkl" % name)


    def dqn_paper_adam_again_noclip_spt():
        lasagne.random.set_rng( np.random.RandomState(0) )
        np.random.seed(0)
        import os
        env = gym.make('Pong-v0')
        env.frameskip = 4
        name = "dqn_paper_revamp2_bn_sgd_noclip_2_rmsprop_bigexperience_spt"
        qq = DeepQ(env, net_fn=architectures.dqn_paper_net_spt, net_fn_args={}, optimiser=rmsprop, optimiser_args={"learning_rate":0.0002, "rho":0.99},
                   img_preprocessor=preprocessor_pong, lambda_fp=0., debug=True, experience_maxlen=500000)
        qq.train(update_q=True, render=True if os.environ["USER"] == "cjb60" else False, min_exploration=-1,
                 max_frames=10000000, save_outfile_to="results/%s" % name, save_weights_to="weights/%s.pkl" % name)
    """

        
    def local():
        lasagne.random.set_rng( np.random.RandomState(0) )
        np.random.seed(0)
        import os
        env = gym.make('Pong-v0')
        name = "local"
        qq = DeepQ(env,
                   net_fn=dqn_paper_net,
                   net_fn_args={},
                   optimiser=nesterov_momentum,
                   optimiser_args={"learning_rate":0.01, "momentum":0.9},
                   img_preprocessor=preprocessor_pong,
                   debug=True,
                   experience_maxlen=100000)
        qq.train(update_q=True,
                 render=True if os.environ["USER"] == "cjb60" else False,
                 max_frames=10000000,
                 save_outfile_to="results/%s.txt" % name,
                 save_weights_to="weights/%s.pkl" % name)


        
    locals()[ sys.argv[1] ]()
        

    """
    for layer in get_all_layers(my_q_net(env)):
        print layer, layer.output_shape
    print count_params(layer)
    """
