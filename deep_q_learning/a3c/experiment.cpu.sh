HDF5_DISABLE_VERSION_CHECK=1 \
THEANO_FLAGS=mode=FAST_RUN,allow_gc=True,floatX=float32,nvcc.fastmath=True,warn_float64=warn \
  python dqn_a3c.py a3c_rmsprop_1worker
