import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
import cPickle as pickle
import gzip
import numpy as np
import multiprocessing

"""
def get_net():
    l_in = InputLayer((None, 784))
    l_dense = DenseLayer(l_in, num_units=256)
    l_dense2 = DenseLayer(l_dense, num_units=128)
    l_softmax = DenseLayer(l_dense2, num_units=10, nonlinearity=softmax)
    return l_softmax
"""

def get_net():
    l_in = InputLayer( (None, 1, 28, 28) )
    l_conv = Conv2DLayer(l_in, num_filters=32, filter_size=3)
    l_mp1 = MaxPool2DLayer(l_conv, pool_size=2)
    l_conv2 = Conv2DLayer(l_mp1, num_filters=48, filter_size=3)
    l_mp2 = MaxPool2DLayer(l_conv2, pool_size=2)
    l_conv3 =  Conv2DLayer(l_mp2, num_filters=64, filter_size=3)
    l_dense = DenseLayer(l_conv3, num_units=10, nonlinearity=softmax)
    return l_dense

with gzip.open("./mnist.pkl.gz") as f:
    train_data, valid_data, _ = pickle.load(f)

X_train, y_train = train_data
X_valid, y_valid = valid_data
X_train = X_train.reshape((X_train.shape[0],1,28,28))
X_valid = X_valid.reshape((X_valid.shape[0],1,28,28))

def iterator(X,y,bs):
    b = 0
    while True:
        if b*bs >= X.shape[0]:
            break
        yield X[(b*bs):(b+1)*bs], y[(b*bs):(b+1)*bs]
        b += 1

import time

def worker(X_train, y_train, net_fn, num_epochs, master_params):
    """
    X_train: the chunk of training data this worker is meant to operate on
    y_train:
    net_fn: network specification
    num_epochs:
    """
    l_out = net_fn()
    X = T.tensor4('X')
    y = T.ivector('y')
    net_out = get_output(l_out, X)
    loss = categorical_crossentropy(net_out, y).mean()
    params = get_all_params(l_out, trainable=True)
    grads = T.grad(loss, params)
    grads_fn = theano.function([X,y], grads)
    close = False
    worker_name = multiprocessing.current_process().name
    print "num epochs", num_epochs
    print len(master_params)
    idxs = [i for i in range(X_train.shape[0])]
    for epoch in range(num_epochs):
        np.random.shuffle(idxs)
        X_train, y_train = X_train[idxs], y_train[idxs]
        print "[%s] epoch: %i" % (worker_name, epoch)
        for X_batch, y_batch in iterator(X_train, y_train, bs=32):
            # 'in the simplest implementation, before processing each minibatch, a model replica
            # asks the parameter server service for an updated copy of its model parameters'
            for i in range(len(params)):
                params[i].set_value( master_params[i] )
            this_grads = grads_fn(X_batch, y_batch)
            for i in range(len(this_grads)):
                master_params[i] = master_params[i] - 0.01*this_grads[i]
    print "[%s] final close..." % worker_name

###########
# workers #
###########

master_params = multiprocessing.Manager().list()

processes = []
num_processes = 4
bs = 2500 # each process takes a chunk of 5000 from the training set
for i in range(num_processes):
    slice_ = slice(i*bs, (i+1)*bs)
    # update params from master process every 3 epochs??
    p = multiprocessing.Process(target=worker,
            args=(X_train[slice_].astype("float32"), y_train[slice_].astype("int32"), get_net, 2000, master_params))
    processes.append(p)
    p.start()

##########
# master #
##########

def master_worker(X_valid, y_valid, net_fn, num_epochs, master_params, eval_every, debug=False):
    t0 = time.time()
    print "starting master worker..."
    l_out = net_fn()
    for layer in get_all_layers(l_out):
        print layer, layer.output_shape
    print count_params(l_out)
    params = get_all_params(l_out, trainable=True)
    for i in range(len(params)):
        master_params.append( params[i].get_value() )
    X = T.tensor4('X')
    y = T.ivector('y')
    net_out = get_output(l_out, X)
    out_fn = theano.function([X], net_out)
    for iter_ in range(num_epochs):
        # all the master worker needs to do is grab the latest master params
        # and then evaluate the accuracy on the validation set
        for i in range(len(params)):
            params[i].set_value( master_params[i] )
        if iter_ % eval_every == 0:
            preds = []
            for X_batch, y_batch in iterator(X_valid, y_valid, bs=32):
                this_preds = np.argmax(out_fn(X_batch),axis=1)
                preds += this_preds.tolist()
            preds = np.asarray(preds)
            valid_acc = (preds == y_valid).mean()
            print "[master]: valid_acc = %f, time taken = %f" % (valid_acc, time.time()-t0)

                
master_worker(
    X_valid[0:1000].astype("float32"),
    y_valid[0:1000].astype("float32"),
    get_net, 10000, master_params, 100)

"""
def debug_worker(X_train, y_train, X_valid, y_valid, net_fn, num_epochs):
    print "starting debug worker..."
    l_out = net_fn()
    for layer in get_all_layers(l_out):
        print layer, layer.output_shape
    print count_params(l_out)
    params = get_all_params(l_out, trainable=True)
    X = T.tensor4('X')
    y = T.ivector('y')
    net_out = get_output(l_out, X)
    out_fn = theano.function([X], net_out)
    loss = categorical_crossentropy(net_out, y).mean()
    grads = T.grad(loss, params)
    grads_fn = theano.function([X,y], grads)
    print X_valid.shape, y_valid.shape
    for iter_ in range(num_epochs):
        print "[debug] epoch %i" % iter_
        for X_batch, y_batch in iterator(X_train, y_train, 32):
            grads = grads_fn(X_batch, y_batch)
            for i in range(len(params)):
                # do sgd on this param
                params[i].set_value( params[i].get_value() - 0.01*grads[i])
        # eval
        preds = []
        for X_batch, _ in iterator(X_valid, y_valid, bs=32):
            this_preds = np.argmax(out_fn(X_batch),axis=1)
            preds += this_preds.tolist()
        preds = np.asarray(preds)
        valid_acc = (preds == y_valid).mean()
        print "[debug]: valid_acc = %f" % valid_acc


       
debug_worker(
    X_train[0:1000].astype("float32"),
    y_train[0:1000].astype("int32"),
    X_valid[0:1000].astype("float32"),
    y_valid[0:1000].astype("int32"), get_net, 10000)
"""










print "exit master worker..."


print "joining processes..."

# problem: the script does not terminate, and I
# am not sure why...
for p in processes:
    p.join()
