HDF5_DISABLE_VERSION_CHECK=1 \
THEANO_FLAGS=mode=FAST_RUN,device=gpu1,allow_gc=True,floatX=float32,nvcc.fastmath=True,warn_float64='warn' \
  python dqn.py dqn_paper_adam_again_noclip_spt
