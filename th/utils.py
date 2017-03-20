import numpy as np
import theano
theano.config.floatX = 'float32'

def ortho_weight(ndim, dtype=theano.config.floatX):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(dtype)

def norm_weight(nin, nout=None, scale=0.01, ortho=False, dtype=theano.config.floatX):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype(dtype)

def plus(pp, name):
    return '%s_%s' % (pp, name)

def itemlist(tparams):
    return [vv for kk, vv in tparams.items()]
