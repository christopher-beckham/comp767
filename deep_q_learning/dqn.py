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
    the buffer, store (x_t,a_t,r_t,x_t1)'s and THEN convert these to the
    (phi_t, ..., phi_t1) format in the DeepQ class during training.
    """
    def __init__(self, maxlen):
        self.experience = []
        self.maxlen = maxlen
        self.counter = 0
        self.debug = True
    def add(self,i):
        if len(self.experience) != self.maxlen:
            self.experience.append(i)
        else:
            self.experience[ self.counter % self.maxlen ] = i
        self.counter += 1
    def sample(self, batch_size):
        len_ = len(self.experience)
        idxs = [idx for idx in range(exp.counter % len_, (exp.counter % len_) + len(exp.experience))]
        pivot = np.random.randint(0, len(idxs)-batch_size+1)
        idxs_pivot = idxs[pivot:(pivot+batch_size)]
        #if self.debug:
        #    assert range(idxs_pivot[0], idxs_pivot[0]+len(idxs_pivot)) == idxs_pivot
        samples = []
        for idx in idxs_pivot:
            samples.append( self.experience[ idx % len_ ] )
        return samples

class MemoryExperienceManager():
    """
    OLD
    """
    def __init__(self, filename, maxlen):
        self.filename = filename
        if os.path.exists(filename):
            self.experience = self.load(filename)
            self.counter = len(self.experience)
        else:
            self.experience = []
            self.counter = 0
        self.experience_maxlen = maxlen
        self.counter = 0
    def write(self, imgs, r_t, a_t, is_done, **kwargs):
        tp = {'imgs':imgs, 'r_t':r_t, 'a_t':a_t, 'is_done':is_done}
        if len(self.experience) != self.experience_maxlen:
            self.experience.append(tp)
        else:
            self.experience[ self.counter % len(self.experience) ] = tp
        self.counter += 1
    def sample(self, batch_size):
        idxs = [x for x in range(len(self.experience))]
        np.random.shuffle(idxs)
        rand_transitions = [ self.experience[idx] for idx in idxs[0:batch_size] ]
        return rand_transitions
    def length(self):
        return len(self.experience)
    def save(self, filename):
        pickle.dump(self.experience, open(filename,"wb"), pickle.HIGHEST_PROTOCOL)
    def load(self, filename):
        return pickle.load(open(filename))

class DiskExperienceManager():
    """
    OLD
    """
    def __init__(self, filename, maxlen, flag, width=80, height=80, nchannels=5, debug=False):
        self.filename = filename
        self.maxlen = maxlen
        # read / append / write new
        assert flag in ["r", "r+", "w+"]
        f = {}
        f['imgs'] = np.memmap("%s.imgs" % filename, mode=flag, shape=(maxlen, 5, height, width), dtype="uint8")
        f['r_t'] = np.memmap("%s.rt" % filename, mode=flag, shape=(maxlen, 1), dtype="float32")
        f['a_t'] = np.memmap("%s.at" % filename, mode=flag, shape=(maxlen, 1), dtype="float32")
        f['is_done'] = np.memmap("%s.id" % filename, mode=flag, shape=(maxlen, 1), dtype="bool")
        f['t'] = np.memmap("%s.t" % filename, mode=flag, shape=(maxlen, 1), 
                           dtype="int32")
        if flag == "w+":
            f['t'][0:maxlen] = np.zeros((maxlen,1),dtype="int32")-1
            self.counter = 0
        else:
            self.counter = np.max(f['t'])+1
            print "self.counter set to: %i" % self.counter
        self.debug = debug
        self.f = f
        self.flag = flag
    def write(self, imgs, r_t, a_t, is_done, **kwargs):
        if self.flag == "r":
            return
        self.f['imgs'][self.counter % self.maxlen] = imgs
        self.f['r_t'][self.counter % self.maxlen] = r_t
        self.f['a_t'][self.counter % self.maxlen] = a_t
        self.f['is_done'][self.counter % self.maxlen] = is_done
        self.f['t'][self.counter % self.maxlen] = self.counter
        self.counter += 1
        for key in self.f:
            self.f[key].flush()
    def sample(self, batch_size):
        t0 = time.time()
        idxs = [i for i in range(0, self.length())]
        np.random.shuffle(idxs)
        chosen_idxs = idxs[0:batch_size]
        experience = []
        for idx in chosen_idxs:
            experience.append(
                {'imgs':np.asarray(self.f['imgs'][idx]),
                 'r_t':np.asarray(self.f['r_t'][idx]).item(),
                 'a_t':np.asarray(self.f['a_t'][idx]).item(),
                 'is_done':np.asarray(self.f['is_done'][idx]).item(),
                 't':np.asarray(self.f['t'][idx]).item()})
            #if self.debug:
            assert self.f['t'][idx] > -1
        print time.time()-t0
        return experience
    def close(self):
        if self.flag == "r":
            return
        for key in self.f:
            self.f[key].flush()
    def length(self):
        return min(self.counter, self.maxlen)

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
        self.l_out_stale = net_fn(self.env, net_fn_args)
        #self.l_out = net_fn(self.env, net_fn_args)
        #self.l_out_stale = net_fn(self.env, net_fn_args)
        #print "q network:"
        self._print_network(self.l_out)
        #print "fp network:"
        #self._print_network(self.l_out_fp)
        self.experience_manager = experience_manager
        self.img_preprocessor = img_preprocessor
        self.debug = debug
        #self.lambda_fp = lambda_fp
        # theano variables for forward pass
        X = T.tensor4('X')
        net_out = get_output(self.l_out, X)
        self.q_fn = theano.function([X], net_out)
        # theano variables for updating Q
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
        # given phi_t, we want it to predict phi_t1
        #output_phi_t1_future = get_output(self.l_out_fp, phi_t)
        td_target = r + (1.0-is_done)*(gamma*T.max(output_phi_t1,axis=1,keepdims=True))
        td_error = (phi_t_mask*output_phi_t).sum(axis=1, keepdims=True)
        if loss == "default":
            loss = squared_error(td_target,td_error).mean()
            #if lambda_fp > 0:
            #    # we want the fp branch to predict phi_t1
            #    print "lambda_fp > 0 so also doing future prediction"
            #    loss += lambda_fp*squared_error(output_phi_t1_future, phi_t1).mean()
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
        #self.fp_fn = theano.function([phi_t], output_phi_t1_future)
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
        if legacy:
            set_all_param_values(self.l_out, weights)
            set_all_param_values(self.l_out_stale, weights)
        else:
            set_all_param_values(self.l_out, weights['q'])
            set_all_param_values(self.l_out_stale, weights['q'])
            set_all_param_values(self.l_out_fp, weights['fp'])
    def _save_weights_to(self, out_file, legacy=True):
        if legacy:
            self._save_as_pickle( get_all_param_values(self.l_out), out_file )
        else:
            self._save_as_pickle({'q': get_all_param_values(self.l_out), 'fp': get_all_param_values(self.l_out_fp)}, out_file)
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
        # sample from random experience from the buffer
        """
        idxs = [i for i in range(0, len(self.experience))] # index into ring buffer
        np.random.shuffle(idxs)
        rand_transitions = \
            [ self.experience[idx] for idx in idxs[0:batch_size] ]
        """
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
            #r_j = 0.
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
                    #if self.debug:
                    #    print "a_t = %i, r_t = %f" % (a_t, r_t)
                    if render:
                        self.env.render()
                    buf.append({'x_t1':self.img_preprocessor(x_t1), 'r_t':r_t, 'a_t':a_t, 't':t})
                    tot_frames += 1
                    # store transition (phi_t, a_t, r_t, phi_t1 into D)
                    if len(buf) == buf_maxlen:
                        phi_t1 = np.asarray([ buf[0]['x_t1'], buf[1]['x_t1'], buf[2]['x_t1'], buf[3]['x_t1'] ] )
                        debug_t1 = [buf[0]['t'], buf[1]['t'], buf[2]['t'], buf[3]['t']]
                        if phi_t != None:
                            tp = {"imgs": np.vstack((phi_t,phi_t1[-1:])),
                                  "r_t":buf[-1]['r_t'],
                                  "a_t":buf[-1]['a_t'],
                                  "is_done":is_done, "debug_t":debug_t, "debug_t1":debug_t1}
                            #self._check_tp(tp)
                            self.experience_manager.write(**tp)
                        phi_t = phi_t1
                        debug_t = debug_t1
                        buf.popleft()
                        if is_done:
                            out_str = "%i,%i,%f,%i,%f,%i,%f" % (ep+1, t+1, np.mean(losses), sum_rewards, curr_eps, self.experience_manager.length(), time.time()-t0)
                            print out_str
                            if f != None:
                                f.write(out_str + "\n"); f.flush()
                            if save_weights_to != None:
                                self._save_weights_to(save_weights_to)
                            #if phi_t != None and self.lambda_fp > 0:
                            #    self._plot_future_predict(phi_t, "%s/fp.%i.png" % (save_outfile_to, ep))
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


        
    def dqn_paper_adam_again_noclip_fp():
        """
        For 'FP': same as above but with future prediction with fp_lambda=1.
        (which seemed to be too much)
        NOTE: since this was run before 03/04, it has the 'off-by-one'
        bug with the tuples. So it might be a good idea to re-run this,
        if time-permitting.
        """
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
