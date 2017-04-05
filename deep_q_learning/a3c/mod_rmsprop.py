import theano
from theano import tensor as T
import numpy as np

from lasagne.updates import get_or_compute_grads
from collections import OrderedDict

"""
These are modified optimisers from Lasagne.
They differ in two ways from the original implementations:
  - Instead of the updates dictionary being in the form
    updates[param] = param - grads*blah, they are in the form:
    updates[param] = grads*blah
  - The updates are split into param_updates and meta_updates,
    where meta updates are for things like the running gradient
    magnitudes in rmsprop.
Because of this setup, you can do something like this:
updates = optimiser(loss, params, **optimiser_args)
grads_fn = theano.function(inputs, updates['param_updates'].values(), updates=updates['meta_updates'])
Effectively, this means that a call to the grads_fn will give you:
  - The gradients, which you can then use to update the params yourself.
  - The necessary meta parameter updates, which is required for optimisers such
    as rmsprop.
"""

def mod_rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    param_updates = OrderedDict()
    meta_updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        meta_updates[accu] = accu_new
        #updates[param] = param - (learning_rate * grad /
        #                          T.sqrt(accu_new + epsilon))
        param_updates[param] = (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))        

    print "param updates", param_updates
    print "meta updates", meta_updates
    return {"param_updates":param_updates, "meta_updates":meta_updates}

def mod_sgd(loss_or_grads, params, learning_rate):
    """
    """
    grads = get_or_compute_grads(loss_or_grads, params)
    param_updates = OrderedDict()
    meta_updates = OrderedDict()

    for param, grad in zip(params, grads):
        #updates[param] = param - learning_rate * grad
        param_updates[param] = learning_rate*grad

    return {"param_updates":param_updates, "meta_updates":meta_updates}
