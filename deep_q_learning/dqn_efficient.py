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

class EfficientMemoryExperienceManager():
    """
    Unlike MemoryExperienceManager, this is meant to be for non-redundant data
    storage. So instead of storing lots of (phi_t, a_t, r_t, phi_t1)'s in
    the buffer, store (x_t,a_t,r_t,x_t1)'s. This means we have two sampling
    methods: _sample() samples the raw tuples (x_t,a_t,r_t,x_t1)'s, and
    sample() puts them into the intermediary format (imgs, a_t, r_t, is_done)
    """
    def __init__(self, maxlen):
        self.experience = []
        self.maxlen = maxlen
        self.counter = 0
        self.debug = True
    def write(self,x_t1,r_t,a_t,is_done,t):
        tp = {'x_t1':x_t1, 'r_t':r_t, 'a_t':a_t, 'is_done':is_done, 't':t}
        if len(self.experience) != self.maxlen:
            self.experience.append(tp)
        else:
            self.experience[ self.counter % self.maxlen ] = tp
        self.counter += 1
    def _sample(self, bs):
        len_ = len(self.experience)
        idxs = [idx for idx in range(self.counter % len_, (self.counter % len_) + len(self.experience))]
        pivot = np.random.randint(0, len(idxs)-bs+1)
        idxs_pivot = idxs[pivot:(pivot+bs)]
        #if self.debug:
        #    assert range(idxs_pivot[0], idxs_pivot[0]+len(idxs_pivot)) == idxs_pivot
        samples = []
        for idx in idxs_pivot:
            samples.append( self.experience[ idx % len_ ] )
        return samples
    def sample(self, batch_size):
        tps = []
        for i in range(batch_size):
            this_sample = self._sample(5)
            # imgs: [x1,x2,x3,x4,x5]
            # a_t = [a4], r_t = [r4]
            tp = {'imgs': [elem['x_t1'] for elem in this_sample],
                  'a_t':this_sample[-1]['a_t'],
                  'r_t':this_sample[-1]['r_t'],
                  'is_done':this_sample[-1]['is_done']}
            tps.append(tp)
        return tps
    def length(self):
        return len(self.experience)

class DeepQ():
    """
    # maybe some interesting notes here??
    # https://www.reddit.com/r/MachineLearning/comments/4hml3e/what_causes_qfunction_to_diverge_and_how_to/ (recommend clipping)
    # https://jaromiru.com/2016/10/21/lets-make-a-dqn-full-dqn/#fn-398-1 (recommend clipping)
    # https://github.com/maciejjaskowski/deep-q-learning/blob/master/dqn.py (huber loss, stale network?)
    # https://computing.ece.vt.edu/~f15ece6504/slides/L26_RL.pdf (stale network)
    # they recommend gradient clipping to [-1, +1]
    """
    def _print_network(self, l_out):
        for layer in get_all_layers(l_out):
            print layer, layer.output_shape
        print "num params: %i" % count_params(layer)
    def __init__(self, env, experience_manager, net_fn, net_fn_args={}, loss="default", optimiser=rmsprop, optimiser_args={"learning_rate":1.0},
                 img_preprocessor=lambda x: x, grad_clip=None, debug=False):
        self.env = env
        self.l_out = net_fn(self.env, net_fn_args)
        print self.l_out
        self.l_out_stale = net_fn(self.env, net_fn_args)
        self._print_network(self.l_out)
        self.experience_manager = experience_manager
        self.img_preprocessor = img_preprocessor
        self.debug = debug
        X = T.tensor4('X')
        net_out = get_output(self.l_out, X)
        self.q_fn = theano.function([X], net_out)
        r = T.fmatrix('r')
        is_done = T.fmatrix('is_done')
        gamma = T.fscalar('gamma')
        phi_t = T.tensor4('phi_t')
        phi_t_mask = T.fmatrix('phi_t_mask')
        phi_t1 = T.tensor4('phi_t1')
        # loss
        assert loss in ["default", "huber"]
        output_phi_t = get_output(self.l_out, phi_t)
        output_phi_t1 = get_output(self.l_out_stale, phi_t1)
        td_target = r + (1.0-is_done)*(gamma*T.max(output_phi_t1,axis=1,keepdims=True))
        td_error = (phi_t_mask*output_phi_t).sum(axis=1, keepdims=True)
        if loss == "default":
            loss = squared_error(td_target,td_error).mean()
        else:
            # experimenting with huber loss here:
            # https://github.com/maciejjaskowski/deep-q-learning/blob/master/dqn.py#L247-L250
            err = td_target - td_error
            quadratic_part = T.minimum(abs(err), 1)
            linear_part = abs(err) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + linear_part
            loss = T.sum(loss)        
        params = get_all_params(self.l_out, trainable=True)
        grads = T.grad(loss, params)
        if grad_clip != None:
            for i in range(len(grads)):
                grads[i] = T.clip(grads[i], grad_clip[0], grad_clip[1])
        updates = optimiser(grads, params, **optimiser_args)
        self.train_fn = theano.function([r, gamma, is_done, phi_t1, phi_t, phi_t_mask], loss, updates=updates)
        self.grads_fn = theano.function([r, gamma, is_done, phi_t1, phi_t, phi_t_mask], grads)
    def _eps_greedy_action(self, phi_t, eps=0.1):
        """
        phi_t: the image for this time step
        """
        if np.random.random() <= eps:
            return self.env.action_space.sample()
        else:
            action_dist = self.q_fn(img_as_float(phi_t[np.newaxis]).astype("float32"))
            if self.debug:
                print "Q(phi_t): %s, argmax Q(phi_t): %s" % (str(action_dist), np.argmax(action_dist,axis=1))
            best_action = np.argmax(action_dist, axis=1)[0]
            return best_action 
    def _save_as_pickle(self, arr, out_file):
        with open(out_file,"w") as f:
            pickle.dump(arr, f, pickle.HIGHEST_PROTOCOL)
    def load_weights_from(self, in_file, legacy=True):
        print "loading weights from: %s" % in_file
        weights = pickle.load(open(in_file))
        set_all_param_values(self.l_out, weights)
        set_all_param_values(self.l_out_stale, weights)
    def _save_weights_to(self, out_file):
        self._save_as_pickle( get_all_param_values(self.l_out), out_file )
    """
    def save_experience_to(self, out_file):
        self._save_as_pickle(self.experience, out_file)
    """
    def _check_tp(self, tp):
        phi_t, phi_t1 = tp["phi_t"], tp["phi_t1"]
        assert np.all( phi_t[1] == phi_t1[0] )
        assert np.all( phi_t[2] == phi_t1[1] )
        assert np.all( phi_t[3] == phi_t1[2] )
    def _sample_from_experience(self, batch_size, gamma):
        rand_transitions = self.experience_manager.sample(batch_size)
        phi_t1_minibatch = \
            img_as_float(
                np.asarray(
                    [ rand_transitions[i]["imgs"][1::] for i in range(len(rand_transitions)) ]
                )
            ).astype("float32")
        r_minibatch = np.asarray(
            [ [rand_transitions[i]["r_t"]] for i in range(len(rand_transitions)) ], dtype="float32")
        is_done_minibatch = np.asarray(
            [ [1.0*rand_transitions[i]["is_done"]] for i in range(len(rand_transitions)) ], dtype="float32")
        # ok, construct Q(phi_t) and its corresponding mask
        phi_t_minibatch = \
            img_as_float(
                np.asarray(
                    [ rand_transitions[i]["imgs"][0:-1] for i in range(len(rand_transitions)) ]
                )
            ).astype("float32")
        mask_t_minibatch = np.zeros((phi_t_minibatch.shape[0], self.env.action_space.n), dtype="float32")
        for i in range(mask_t_minibatch.shape[0]):
            mask_t_minibatch[ i, rand_transitions[i]["a_t"] ] = 1.
        #import pdb
        #pdb.set_trace()
        #print r_minibatch
        #print phi_t1_minibatch
        #print phi_t_minibatch
        #print mask_t_minibatch
        #print [ rand_transitions[i]["a_t"] for i in range(len(rand_transitions)) ]
        #print "----"
        #self.train_fn = theano.function([r, gamma, is_done, phi_t1, phi_t, phi_t_mask], loss, updates=updates, on_unused_input='warn')
        return r_minibatch, np.float32(gamma), is_done_minibatch, phi_t1_minibatch, phi_t_minibatch, mask_t_minibatch

    def _mkdir_if_not_exist(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def _plot_future_predict(self, phi_t, out_file):
        plt.figure(figsize=(10,6))
        for i in range(4):
            plt.subplot(2,4,i+1)
            plt.imshow(phi_t[i])
        for i in range(4):
            plt.subplot(2,4,4+i+1)
            phi_t1_predicted = self.fp_fn(img_as_float(phi_t)[np.newaxis].astype("float32"))
            plt.imshow(phi_t1_predicted[0,i])
        plt.savefig(out_file)
    
    def train(self,
              render=False,
              gamma=0.95,
              eps_max=1.0,
              eps_min=0.1,
              eps_thresh=1000000.0,
              min_exploration=500000.0,
              max_frames=1000,
              update_q=True,
              batch_size=32,
              save_outfile_to=None,
              save_weights_to=None,
              update_stale_net_every=10000,
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
        self._mkdir_if_not_exist(save_outfile_to)
        out_file = "%s/results.txt" % save_outfile_to
        f_flag = "a" if os.path.exists(out_file) else "wb"
        f = open(out_file, f_flag) if save_outfile_to != None else None
        f.write("episode,num_iters,loss,sum_rewards,curr_eps,time\n")
        tot_frames = 0
        eps_dec_factor = (eps_max - eps_min) / eps_thresh
        curr_eps = eps_max
        t0 = time.time()
        for ep in itertools.count():
            losses = []
            sum_rewards = 0.
            buf_maxlen = 4
            buf = deque(maxlen=buf_maxlen)
            self.env.reset()
            a_t = self.env.action_space.sample()
            phi_t, debug_t = None, None
            for t in itertools.count():
                try:
                    if update_q and tot_frames > min_exploration:
                        if tot_frames <= eps_thresh:
                            curr_eps = eps_max - (tot_frames*eps_dec_factor)
                        else:
                            curr_eps = eps_min
                    if phi_t != None:
                        a_t = self._eps_greedy_action(phi_t, eps=curr_eps)
                    else:
                        a_t = a_t
                    # execute action a_t in emulator and observe
                    # reward r_t and image x_t+1
                    x_t1, r_t, is_done, info = self.env.step(a_t)
                    sum_rewards += r_t
                    if render:
                        self.env.render()
                    tp = {'x_t1':self.img_preprocessor(x_t1), 'r_t':r_t, 'a_t':a_t, 'is_done':is_done, 't':t}
                    buf.append(tp)
                    self.experience_manager.write(**tp)
                    tot_frames += 1
                    # store transition (phi_t, a_t, r_t, phi_t1 into D)
                    if len(buf) == buf_maxlen:
                        phi_t = np.asarray([elem['x_t1'] for elem in buf])
                        debug_t = [ elem['t'] for elem in buf ]
                        buf.popleft()
                        if is_done:
                            out_str = "%i,%i,%f,%i,%f,%i,%f" % (ep+1, t+1, np.mean(losses), sum_rewards, curr_eps, self.experience_manager.length(), time.time()-t0)
                            print out_str
                            if f != None:
                                f.write(out_str + "\n"); f.flush()
                            if save_weights_to != None:
                                self._save_weights_to(save_weights_to)
                            break

                    if self.experience_manager.length() > batch_size and update_q:
                        r_minibatch, gamma_, is_done_minibatch, phi_t1_minibatch, phi_t_minibatch, mask_t_minibatch = \
                            self._sample_from_experience(batch_size, gamma)
                        #self.train_fn = theano.function([r, gamma, is_done_minibatch, phi_t1, phi_t, phi_t_mask], loss, updates=updates, on_unused_input='warn')
                        this_loss = self.train_fn(
                            r_minibatch, gamma_, is_done_minibatch, phi_t1_minibatch, phi_t_minibatch, mask_t_minibatch)
                        losses.append(this_loss)
                        if tot_frames % update_stale_net_every == 0:
                            print "updating stale network with params of main network"
                            set_all_param_values( self.l_out_stale, get_all_param_values(self.l_out) )

                    if tot_frames >= max_frames:
                        return
                    
                except KeyboardInterrupt:
                    import pdb
                    pdb.set_trace()


def preprocessor_pong(img):
    img = rgb2gray(img) # (210, 160)
    img = resize(img, (img.shape[0]//2, img.shape[1]//2)) # (105, 80)
    img = img[17:97,:] # (80, 80)
    img = img_as_ubyte(img) # to save on memory
    return img
                    
if __name__ == '__main__':
    pass
