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
import cPickle as pickle
import itertools
from collections import deque

def my_q_net(env):
    height, width, nchannels = env.observation_space.shape
    nchannels = 4 # we convert to black and white and use 4 prev frames
    l_in = InputLayer((None, nchannels, height, width))
    l_conv = Conv2DLayer(l_in, num_filters=32, filter_size=3, stride=2)
    l_conv2 = Conv2DLayer(l_conv, num_filters=64, filter_size=3, stride=2)
    l_conv3 = Conv2DLayer(l_conv2, num_filters=96, filter_size=3, stride=2)
    l_conv4 = Conv2DLayer(l_conv3, num_filters=128, filter_size=3, stride=2)
    l_dense = DenseLayer(l_conv4, num_units=env.action_space.n)
    return l_dense

def dqn_paper_net(env, args={}):
    nonlinearity = rectify if "nonlinearity" not in args else args["nonlinearity"]
    height, width, nchannels = env.observation_space.shape
    nchannels = 4 # we convert to black and white and use 4 prev frames
    layer = InputLayer((None, nchannels, height, width))
    layer = Conv2DLayer(layer, filter_size=8, num_filters=16, stride=4, nonlinearity=nonlinearity)
    layer = Conv2DLayer(layer, filter_size=4, num_filters=32, stride=2, nonlinearity=nonlinearity)
    layer = DenseLayer(layer, num_units=256, nonlinearity=nonlinearity)
    layer = DenseLayer(layer, num_units=env.action_space.n, nonlinearity=linear)
    return layer

class DeepQ():
    def __init__(self, env, net_fn, net_fn_args={}, optimiser=rmsprop, optimiser_args={"learning_rate":1.0}, experience_maxlen=20000, debug=False):
        self.env = env
        #self.l_out = self._q_net(self.env)
        self.l_out = net_fn(self.env, net_fn_args)
        self.experience_maxlen = experience_maxlen
        self.experience = []
        self.debug = debug
        # theano variables
        X = T.tensor4('X')
        y = T.fmatrix('y')
        action_mask = T.fmatrix('action_mask')
        # loss
        net_out = get_output(self.l_out, X)
        loss = ( y - (action_mask*net_out).sum(axis=1, keepdims=True) )**2
        loss = loss.mean()
        # theano functions
        self.q_fn = theano.function([X], net_out)
        params = get_all_params(self.l_out, trainable=True)
        updates = optimiser(loss, params, **optimiser_args)
        self.train_fn = theano.function([X,y,action_mask], loss, updates=updates)
    def _preprocess_frame(self, img):
        return rgb2gray(img)
    def _eps_greedy_action(self, phi_t, eps=0.1):
        """
        phi_t: the pre-processed image for this time step
        """
        if np.random.random() <= eps:
            return self.env.action_space.sample()
        else:
            action_dist = self.q_fn(phi_t[np.newaxis])
            if self.debug:
                print "Q(phi_t): %s, argmax Q(phi_t): %s" % (str(action_dist), np.argmax(action_dist,axis=1))
            best_action = np.argmax(action_dist, axis=1)[0]
            return best_action 
    def _save_as_pickle(self, arr, out_file):
        with open(out_file,"w") as f:
            pickle.dump(arr, f, pickle.HIGHEST_PROTOCOL)
    def load_weights_from(self, in_file):
        print "loading weights from: %s" % in_file
        weights = pickle.load(open(in_file))
        set_all_param_values(self.l_out, weights)   
    def _save_weights_to(self, out_file):
        self._save_as_pickle(get_all_param_values(self.l_out), out_file)
    def save_experience_to(self, out_file):
        self._save_as_pickle(self.experience, out_file)
    def _sample_from_experience(self, batch_size, gamma):
        # sample from random experience from the buffer
        idxs = [i for i in range(0, len(self.experience))] # index into ring buffer
        np.random.shuffle(idxs)
        rand_transitions = \
            [ self.experience[idx] for idx in idxs[0:batch_size] ]
        # ok, construct the target y_j, which is:
        # r_j + gamma*max_a' Q(phi_j+1)
        phi_t1_minibatch = np.asarray(
            [ rand_transitions[i]["phi_t1"] for i in range(len(rand_transitions)) ], dtype="float32")
        qvalues_t1_minibatch = self.q_fn(phi_t1_minibatch)
        max_qvalues_t1_minibatch = np.max(qvalues_t1_minibatch,axis=1)
        y_minibatch = []
        for i in range(qvalues_t1_minibatch.shape[0]):
            if rand_transitions[i]["is_done"]:
                y_minibatch.append([rand_transitions[i]["r_t"]])
            else:
                y_minibatch.append([rand_transitions[i]["r_t"]+gamma*max_qvalues_t1_minibatch[i]])
        y_minibatch = np.asarray(y_minibatch, dtype="float32")
        # ok, construct Q(phi_t) and its corresponding mask
        phi_t_minibatch = np.asarray(
            [ rand_transitions[i]["phi_t"] for i in range(len(rand_transitions)) ], dtype="float32")
        mask_t_minibatch = np.zeros((phi_t_minibatch.shape[0], self.env.action_space.n), dtype="float32")
        for i in range(mask_t_minibatch.shape[0]):
            mask_t_minibatch[ i, rand_transitions[i]["a_t"] ] = 1.
        #print mask_minibatch
        #print "qvalues_minibatch", qvalues_minibatch.shape
        #print "y_minibatch", y_minibatch.shape
        #print "mask_minibatch", mask_minibatch.shape
        #print "phi_t1_minibatch", phi_t1_minibatch.shape
        return phi_t_minibatch, mask_t_minibatch, y_minibatch
        
    def train(self,
              render=False,
              gamma=0.95,
              eps=1.,
              max_frames=1000,
              update_q=True,
              batch_size=32,
              save_outfile_to=None,
              save_weights_to=None,
              debug=False):
        """
        :render: do we render the game?
        :gamma: discount factor
        :eps: initial eps factor for exploration
        :max_frames: maximum # of frames to see before termination of this fn
        :update_q: backprop through Q fn
        :batch_size: batch size for updates
        :save_outfile_to: save outfile to
        :experience_maxlen
        :debug:
        """
        f = open(save_outfile_to, "wb") if save_outfile_to != None else None
        tot_frames = 0
        eps_max, eps_min, thresh = eps, 0.1, 1000000.0
        eps_dec_factor = (eps_max - eps_min) / thresh
        curr_eps = eps_max
        for ep in itertools.count():
            losses = []
            buf_maxlen = 4
            buf = deque(maxlen=buf_maxlen)
            self.env.reset()
            a_t = self.env.action_space.sample()
            phi_t = None
            for t in itertools.count():
                # with probability eps, select a random action a_t
                # otherwise select it from the Q function
                # but we only want to do this every k'th frame (where k = 4)
                # in this case
                if (t+1) % buf_maxlen == 0 and phi_t != None:
                    # from deep-q paper: anneal eps from 1 to 0.1
                    # over the course of 1m iterations
                    if update_q:
                        if tot_frames <= thresh:
                            curr_eps = eps_max - (tot_frames*eps_dec_factor)
                        else:
                            curr_eps = eps_min
                    a_t = self._eps_greedy_action(phi_t, eps=curr_eps)
                else:
                    a_t = a_t
                # execute action a_t in emulator and observe
                # reward r_t and image x_t+1
                x_t1, r_t, is_done, info = self.env.step(a_t)
                #if self.debug:
                #    print "a_t = %i, r_t = %f" % (a_t, r_t)
                if render:
                    self.env.render()
                buf.append(self._preprocess_frame(x_t1))
                tot_frames += 1
                #print len(buf)
                # store transition (phi_t, a_t, r_t, phi_t1 into D)
                # NOTE: this requires two conditions to be met:
                # - that phi_t != None (it is None on the first execution of the
                #   below if statement)
                # - that phi_t exists (which is only the case when the buffer is full)
                if len(buf) == buf_maxlen:
                    phi_t1 = np.asarray(list(buf), dtype="float32")
                    if phi_t != None:
                        tp = {"phi_t":phi_t, "a_t":a_t, "r_t":r_t, "phi_t1":phi_t1, "is_done":is_done}
                        if len(self.experience) != self.experience_maxlen:
                            self.experience.append(tp)
                        else:
                            self.experience[ tot_frames % len(self.experience) ] = tp   
                    phi_t = phi_t1
                    buf.popleft() # e.g. if buffer was [1,2,3,4], it will now be [2,3,4], so we can add [5] to it next iterationB
                    if is_done:
                        out_str = "episode %i took %i iterations, avg loss = %f, curr_eps = %f" % \
                            (ep+1, t+1, np.mean(losses), curr_eps)
                        print out_str
                        if f != None:
                            f.write(out_str + "\n"); f.flush()
                        if save_weights_to != None:
                            self._save_weights_to(save_weights_to)
                        break

                    if len(self.experience) > batch_size and update_q:
                        phi_t_minibatch, mask_t_minibatch, y_minibatch = self._sample_from_experience(batch_size, gamma)
                        this_loss = self.train_fn(phi_t_minibatch, y_minibatch, mask_t_minibatch)
                        losses.append(this_loss)
                    
                if tot_frames >= max_frames:
                    #return experience, losses
                    return
                    
            #return losses
        #return experience, losses
        return

if __name__ == '__main__':

    import sys

    def dqn_paper_1():
        # rmsprop by default
        # loss blows up at the start and doesn't seem to decrease, w/ lr = 1.0 and 0.1
        # (maybe a lower one will do?)
        env = gym.make('Pong-v0')
        qq = DeepQ(env, net_fn=dqn_paper_net, debug=True)
        #qq.load_weights_from("./weights.pkl.test")
        qq.train(update_q=True, max_frames=10000000, eps=1., save_outfile_to="dqn_paper_fixed_lr0.1.txt", save_weights_to="dqn_paper_weights_fixed_lr0.1.pkl")

    def dqn_paper_adam():
        env = gym.make('Pong-v0')
        qq = DeepQ(env, net_fn=dqn_paper_net, optimiser=adam, optimiser_args={}, debug=True)
        #qq.load_weights_from("./weights.pkl.test")
        qq.train(update_q=True, max_frames=10000000, eps=1., save_outfile_to="dqn_paper_fixed_adam.txt", save_weights_to="dqn_paper_weights_fixed_adam.pkl")
        
    def dqn_paper_leakyrelu():
        env = gym.make('Pong-v0')
        qq = DeepQ(env, net_fn=dqn_paper_net, net_fn_args={}, debug=True)
        qq.train(update_q=True, max_frames=10000000, eps=1., save_outfile_to="dqn_paper_2.txt", save_weights_to="dqn_paper_2_weights")

    def local_test():
        env = gym.make('Pong-v0')
        qq = DeepQ(env, net_fn=dqn_paper_net, net_fn_args={}, debug=True)
        qq.load_weights_from("dqn_paper_weights_fixed_adam.pkl.bak")
        qq.train(update_q=False, max_frames=1000000, eps=0.1, render=True)
        #qq.save_experience_to("test_exp.pkl")
        

    locals()[ sys.argv[1] ]()
        

    """
    for layer in get_all_layers(my_q_net(env)):
        print layer, layer.output_shape
    print count_params(layer)
    """
