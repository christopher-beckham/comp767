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

def get_net():
    l_in = InputLayer((None, 784))
    l_dense = DenseLayer(l_in, num_units=256)
    l_dense2 = DenseLayer(l_dense, num_units=128)
    l_softmax = DenseLayer(l_dense2, num_units=10, nonlinearity=softmax)
    return l_softmax

with gzip.open("./mnist.pkl.gz") as f:
    train_data, valid_data, _ = pickle.load(f)

X_train, y_train = train_data
X_valid, y_valid = valid_data

def iterator(X,y,bs):
    b = 0
    while True:
        if b*bs >= X.shape[0]:
            break
        yield X[(b*bs):(b+1)*bs], y[(b*bs):(b+1)*bs]
        b += 1

import time

def worker(X_train, y_train, net_fn, num_epochs, queue, master_params, update_every):
    """
    X_train: the chunk of training data this worker is meant to operate on
    y_train:
    net_fn: network specification
    num_epochs:
    queue: this worker puts its list of gradients in here for the master worker
      to consume
    """
    l_out = net_fn()
    X = T.fmatrix('X')
    y = T.ivector('y')
    net_out = get_output(l_out, X)
    loss = categorical_crossentropy(net_out, y).mean()
    params = get_all_params(l_out, trainable=True)
    grads = T.grad(loss, params)
    grads_fn = theano.function([X,y], grads)
    close = False
    worker_name = multiprocessing.current_process().name
    print "num epochs", num_epochs
    for epoch in range(num_epochs):
        if epoch % update_every == 0:
            print "[%s] updating params..." % worker_name
            #  UPDATE PARAMS
            for i in range(len(params)):
                params[i].set_value( master_params[i] )
        print "[%s] epoch: %i" % (worker_name, epoch)
        for X_batch, y_batch in iterator(X_train, y_train, bs=32):
            # this breaks the epoch for loop
            this_grads = grads_fn(X_batch, y_batch)
            #print "[%s] queue empty =" % worker_name, queue.empty(), "queue full =", queue.full()
            try:
                queue.put(this_grads, timeout=2)
            except:
                print "[%s] timeout for queue.put, so exiting..." % worker_name
                close = True
                break
        if close:
            print "[%s] closing...." % worker_name
            break
    print "[%s] final close..." % worker_name

###########
# workers #
###########

queue = multiprocessing.Queue(maxsize=10)
master_params = multiprocessing.Manager().list()

p = multiprocessing.Process(target=worker, 
        args=(X_train[0:100].astype("float32"), y_train[0:100].astype("int32"), get_net, 2000, queue, master_params, 5))
p.start()

##########
# master #
##########

def master_worker(X_valid, y_valid, net_fn, num_epochs, queue, master_params):
    print "starting master worker..."
    l_out = net_fn()
    params = get_all_params(l_out, trainable=True)
    for i in range(len(params)):
        master_params.append( params[i].get_value() )
    X = T.fmatrix('X')
    y = T.ivector('y')
    net_out = get_output(l_out, X)
    out_fn = theano.function([X], net_out)
    for iter_ in range(num_epochs):
        print "[master]: epoch %i" % iter_
        # ok, let's try and get a grad object from the queue
        # and then update our params before evaluating on
        # the validation set
        #print "[master] queue empty =", queue.empty(), "queue full =", queue.full()
        grads = queue.get()
        print "got grads, doing an update..."
        for i in range(len(params)):
            # do sgd on this param
            params[i].set_value( params[i].get_value() - 0.01*grads[i])    
        preds = []
        for X_batch, y_batch in iterator(X_valid, y_valid, bs=32):
            this_preds = np.argmax(out_fn(X_batch),axis=1)
            preds += this_preds.tolist()
        preds = np.asarray(preds)
        valid_acc = (preds == y_valid).mean()
        print "[master]: valid_acc = %f" % valid_acc
        # UPDATE PARAMS
        for i in range(len(params)):
            master_params[i] = params[i].get_value()

master_worker(X_valid[0:100].astype("float32"), y_valid[0:100].astype("float32"), get_net, 100, queue, master_params)

print "exit master worker..."

#while queue.full():
#    # consume the entire queue
#    queue.get()

# wait for the worker thread to finish


print "joining queue"

queue.close()
queue.join_thread()

print "joining p"

"""
while queue.full():
    print "queue is full so flushing it..."
    queue.get()
"""

# problem: the script does not terminate, and I
# am not sure why...
p.join()
