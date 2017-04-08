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
from mod_rmsprop import mod_sgd, mod_rmsprop # MODIFIED OPTIMISERS
import time

class DeepQ():
    def _print_network(self, l_out):
        for layer in get_all_layers(l_out):
            print layer, layer.output_shape
        print "num params: %i" % count_params(layer)
    def __init__(self, env, net_fn, net_fn_args={}, loss="default", optimiser=mod_sgd, optimiser_args={"learning_rate":0.01},
                 img_preprocessor=lambda x: x, grad_clip=None, experience_maxlen=20000, debug=False, name=""):
        self.env = env
        self.l_out = net_fn(self.env, net_fn_args)
        self.l_out_stale = net_fn(self.env, net_fn_args)
        print "q network:"
        self._print_network(self.l_out)
        self.img_preprocessor = img_preprocessor
        self.debug = debug
        #self.learning_rate = learning_rate
        self.name = name
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
        #grads = T.grad(loss, params)
        updates = optimiser(loss, params, **optimiser_args)
        #if grad_clip != None:
        #    for i in range(len(grads)):
        #        grads[i] = T.clip(grads[i], grad_clip[0], grad_clip[1])
        if grad_clip != None:
            raise NotImplementedError()
        #updates = optimiser(grads, params, **optimiser_args)
        #self.grads_fn = theano.function([r, gamma, is_done, phi_t1, phi_t, phi_t_mask], grads)
        self.grads_fn = theano.function([r, gamma, is_done, phi_t1, phi_t, phi_t_mask],
                                        updates["param_updates"].values(), updates=updates["meta_updates"])
        self.loss_fn = theano.function([r, gamma, is_done, phi_t1, phi_t, phi_t_mask], loss)
        self.params = params
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
        set_all_param_values(self.l_out_stale, weights)
    def _save_weights_to(self, out_file):
        self._save_as_pickle( get_all_param_values(self.l_out), out_file )
    def _check_tp(self, tp):
        phi_t, phi_t1 = tp["phi_t"], tp["phi_t1"]
        assert np.all( phi_t[1] == phi_t1[0] )
        assert np.all( phi_t[2] == phi_t1[1] )
        assert np.all( phi_t[3] == phi_t1[2] )
    def _generate_inputs_from_tp(self, tp, gamma):
        phi_t1_minibatch = img_as_float(np.asarray([ tp["imgs"][1::] ])).astype("float32")
        r_minibatch = np.asarray([ [tp["r_t"]] ], dtype="float32")
        is_done_minibatch = np.asarray([ [1.0*tp["is_done"]] ], dtype="float32")
        phi_t_minibatch = img_as_float(np.asarray([ tp["imgs"][0:-1] ])).astype("float32")
        mask_t_minibatch = np.zeros((1, self.env.action_space.n), dtype="float32")
        mask_t_minibatch[ 0, tp["a_t"] ] = 1.
        return r_minibatch, np.float32(gamma), is_done_minibatch, phi_t1_minibatch, phi_t_minibatch, mask_t_minibatch

    def _mkdir_if_not_exist(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    
    def train(self, master_params, render=False, gamma=0.95,
              eps_max=1.0, eps_min=0.1, eps_thresh=1000000.0, max_frames=1000, worker=True,
              save_outfile_to=None, save_weights_to=None, update_stale_net_every=10000, debug=False):
        """
        TODO: get rid of experience replay. a3c does not use this
        :render: do we render
        :gamma:
        :eps_max:
        :eps_min:
        :eps_thresh:
        :max_frames:
        :worker: if true, this will update `master_params`
        :batch_size:
        :save_outfile_to:
        :save_weights_to:
        :update_stale_net_every:
        """
        def get_empty_grads():
            accumulate_grads = []
            for i in range(len(self.params)):
                accumulate_grads.append( np.zeros_like(self.params[i].get_value()) )
            return accumulate_grads
        # NOTE: doesn't seem to work when > 1
        I_ASYNC_UPDATE = 1 # how often does the worker update the master params with its params
        #I_MASTER = 10 # how often does the master/worker update its params with the master params
        HEADER_WORKER = "episode,num_iters,loss,sum_rewards,curr_eps,time"
        HEADER_MASTER = "episode,num_iters,sum_rewards,curr_eps,time"
        if save_outfile_to != None:
            self._mkdir_if_not_exist(save_outfile_to)
        if save_weights_to != None:
            self._mkdir_if_not_exist(save_weights_to)
        outfile_path = "%s/results.txt" % save_outfile_to
        f_flags = "a" if os.path.isfile(outfile_path) else "wb"
        f = open(outfile_path, f_flags) if save_outfile_to != None else None
        if worker:
            f.write(HEADER_WORKER + "\n")
        else:
            f.write(HEADER_MASTER + "\n")
        tot_frames = 0
        eps_dec_factor = (eps_max - eps_min) / eps_thresh
        curr_eps = eps_max
        accumulate_grads = get_empty_grads()
        if not worker:
            # the master thread initially populates the master params
            # list with its own params
            for i in range(len(self.params)):
                master_params.append( self.params[i].get_value() )
        t0 = time.time()
        for ep in itertools.count():
            #if debug and not worker:
            #    print "master worker update: params checksum:", [ np.sum(self.params[i].get_value()**2) for i in range(len(self.params)) ]
            losses = []
            sum_rewards = 0.
            buf_maxlen = 4
            buf = deque(maxlen=buf_maxlen)
            self.env.reset()
            a_t = self.env.action_space.sample()
            phi_t, debug_t = None, None
            for t in itertools.count():
                try:
                    # for both worker and master, we want to have
                    # the most up to date parameters, so update at
                    # every time step. this is because each thread
                    # operates on \theta, which is master_params in our case
                    if len(self.params) == len(master_params):
                        for i in range(len(self.params)):
                            self.params[i].set_value( master_params[i] )
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
                    buf.append({'x_t1':self.img_preprocessor(x_t1), 'r_t':r_t, 'a_t':a_t, 't':t})
                    tot_frames += 1
                    if len(buf) == buf_maxlen:
                        phi_t1 = np.asarray([ buf[0]['x_t1'], buf[1]['x_t1'], buf[2]['x_t1'], buf[3]['x_t1'] ] )
                        debug_t1 = [buf[0]['t'], buf[1]['t'], buf[2]['t'], buf[3]['t']]
                        if phi_t != None:
                            tp = {"imgs": np.vstack((phi_t,phi_t1[0:1])),
                                  "r_t":buf[-1]['r_t'],
                                  "a_t":buf[-1]['a_t'],
                                  "is_done":is_done, "debug_t":debug_t, "debug_t1":debug_t1}
                            if worker:
                                # do accum grads
                                r_minibatch, gamma_, is_done_minibatch, phi_t1_minibatch, phi_t_minibatch, mask_t_minibatch = \
                                    self._generate_inputs_from_tp(tp, gamma)
                                this_grads = self.grads_fn(
                                    r_minibatch, gamma_, is_done_minibatch, phi_t1_minibatch, phi_t_minibatch, mask_t_minibatch)
                                this_loss = self.loss_fn(
                                    r_minibatch, gamma_, is_done_minibatch, phi_t1_minibatch, phi_t_minibatch, mask_t_minibatch)
                                losses.append(this_loss)
                                # accumulate the gradients
                                for i in range(len(accumulate_grads)):
                                    accumulate_grads[i] += this_grads[i]                                
                        phi_t = phi_t1
                        debug_t = debug_t1
                        buf.popleft()
                        if is_done:
                            if worker:
                                out_str = "%i,%i,%f,%i,%f,%f" % (ep+1, t+1, np.mean(losses), sum_rewards, curr_eps, time.time()-t0)
                            else:
                                out_str = "%i,%i,%i,%f,%f" % (ep+1, t+1, sum_rewards, curr_eps, time.time()-t0)
                            print self.name + " " + out_str
                            if f != None:
                                f.write(out_str + "\n"); f.flush()
                            if save_weights_to != None:
                                self._save_weights_to("%s/weights.pkl" % save_weights_to)
                            break

                    if worker:
                        if tot_frames % I_ASYNC_UPDATE == 0:
                            # every so often, update the master params with our
                            # accumulated gradients, then clear the accumulated grads.
                            # this is the I_async_update part of the algorithm in the paper.
                            if len(master_params) == len(self.params):
                                for i in range(len(accumulate_grads)):
                                    master_params[i] = master_params[i] - accumulate_grads[i]
                                    #if i == 0:
                                    #    print "time %i, worker update master params: accumulate checksum: %s" % \
                                    #        (tot_frames,[ np.sum(accumulate_grads[i]**2) for i in range(len(accumulate_grads))])
                                accumulate_grads = get_empty_grads()
                        
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
    import os
    import multiprocessing

    def preprocessor_pong(img):
        img = rgb2gray(img) # (210, 160)
        img = resize(img, (img.shape[0]//2, img.shape[1]//2)) # (105, 80)
        img = img[17:97,:] # (80, 80)
        img = img_as_ubyte(img) # to save on memory
        return img

    #optimiser=rmsprop,
    #optimiser_args={"learning_rate":.0002, "rho":0.99},
    
    def a3c_rmsprop_1worker():
        """
        """
        lasagne.random.set_rng( np.random.RandomState(0) )
        np.random.seed(0)
        env = gym.make('Pong-v0')
        env.frameskip = 4
        num_processes = 4
        name = "a3c_rmsprop_%iworker_nobuf" % num_processes
        master_params = multiprocessing.Manager().list()
        default_params = {
            "env":env,
            "net_fn":architectures.dqn_paper_net, "net_fn_args":{},
            "optimiser": mod_rmsprop, "optimiser_args":{"learning_rate":0.0002, "rho":0.99},
            "img_preprocessor":preprocessor_pong,
            "debug":False,
            "experience_maxlen":200000
        }
        def worker(process_name):
            worker_params = default_params.copy()
            worker_params["name"] = process_name
            qq = DeepQ(**worker_params)
            qq.train(worker=True,
                     master_params=master_params,
                     render=True if os.environ["USER"] == "cjb60" else False,
                     max_frames=10000000,
                     save_outfile_to="results/%s/%s" % (name, process_name),
                     save_weights_to=None,
                     debug=True)
        processes = []
        for i in range(num_processes):
            p = multiprocessing.Process(target=worker, args=("p%i" % (i+1),))
            processes.append(p)
            p.start()
        #worker("p1")
        def master(process_name):
            m_params = default_params.copy()
            m_params["name"] = process_name
            qq = DeepQ(**m_params)
            qq.train(worker=False,
                     master_params=master_params,
                     render=True if os.environ["USER"] == "cjb60" else False,
                     max_frames=1000000000,
                     save_outfile_to="results/%s/%s" % (name, process_name),
                     save_weights_to="weights/%s/%s" % (name, process_name),
                     debug=True)
        master("master")


        

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
