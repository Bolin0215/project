import theano
import theano.tensor as T
from th.utils import *


def param_init_lstm(params, dim_in, dim_hid, prefix):
    '''
    i = g(wx + uh + b)
    g = g(wx + uh + b)
    o = g(wx + uh + b)
    c1 = tanh(wx + uh + b)
    c = f * c_old + i * c1
    h = o * tanh(c)
    '''
    W = np.concatenate([norm_weight(dim_in, dim_hid),
                        norm_weight(dim_in, dim_hid),
                        norm_weight(dim_in, dim_hid),
                        norm_weight(dim_in, dim_hid)], axis=1)
    params[plus(prefix, 'W')] = W

    U = np.concatenate([norm_weight(dim_hid, dim_hid),
                        norm_weight(dim_hid, dim_hid),
                        norm_weight(dim_hid, dim_hid),
                        norm_weight(dim_hid, dim_hid)], axis=1)
    params[plus(prefix, 'U')] = U

    b = np.zeros((4 * dim_hid,))
    params[plus(prefix, 'b')] = b.astype(theano.config.floatX)

    return params


def lstm_layer(tparams, inputs, nin, ndim, mask=None, prefix='lstm'):
    '''
    :param tparams: shared variables
    :param inputs: [N, M, nin], max_num_words*batch_size*nin or [N, nin]
    :param mask: [N, M]
    :return: [N, M, ndim]
    '''
    nsteps = inputs.shape[0]
    if inputs.ndim == 3:
        nsamples = inputs.shape[1]
    else:
        nsamples = 1
    # assert ndim == tparams[plus(prefix, 'U')].shape[1]
    if mask is None:
        mask = T.alloc(1., inputs.shape[0], 1)

    def _slice(_x, n, dim):
        return _x[:, n * dim: (n + 1)*dim]

    state_below = T.dot(inputs, tparams[plus(prefix, 'W')]) + tparams[plus(prefix, 'b')]

    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, tparams[plus(prefix, 'U')])
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, ndim))
        f = T.nnet.sigmoid(_slice(preact, 1, ndim))
        o = T.nnet.sigmoid(_slice(preact, 2, ndim))
        c = T.tanh(_slice(preact, 3, ndim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[T.alloc(0., nsamples, ndim),
                                              T.alloc(0., nsamples, ndim)],
                                name=plus(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]

def param_init_dyAtt(params, dim_in, dim_hid, prefix='dyAtt'):
    # https://arxiv.org/pdf/1509.06664.pdf
    Wk = norm_weight(dim_in, dim_hid)
    params[plus(prefix, 'Wk')] = Wk

    Wh = norm_weight(dim_in, dim_hid)
    params[plus(prefix, 'Wh')] = Wh

    Wr = norm_weight(dim_hid, dim_hid)
    params[plus(prefix, 'Wr')] = Wr

    W_att = norm_weight(dim_hid, 1)
    params[plus(prefix, 'W_att')] = W_att

    Wt = norm_weight(dim_hid, dim_hid)
    params[plus(prefix, 'Wt')] = Wt

    return params

def dyAtt_layer(tparams, inputs, context, context_mask=None, pad_mask=None, prefix='dyAtt'):
    nsteps = inputs.shape[0]
    if inputs.ndim == 3:
        nsamples = inputs.shape[1]
    else:
        nsamples = 1

    ndim = tparams[plus(prefix, 'Wk')].shape[1]

    def _step(h_, r_, c_):
        Mt_l = T.dot(c_, tparams[plus(prefix, 'Wk')])
        Mt_r = T.dot(h_, tparams[plus(prefix, 'Wh')]) + T.dot(r_, tparams[plus(prefix, 'Wr')])
        Mt = T.tanh(Mt_l + Mt_r)

        alpha = T.dot(Mt, tparams[plus(prefix, 'W_att')])
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = T.exp(alpha)
        if pad_mask:
            alpha = alpha * pad_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx = (c_ * alpha[:, :, None]).sum(0)

        r_l = T.tanh(T.dot(r_, tparams[plus(prefix, 'Wt')]))
        r_ = ctx + r_l
        return r_

    rval, updates = theano.scan(_step,
                                sequences=[inputs],
                                outputs_info=[T.alloc(0., nsamples, ndim)],
                                non_sequences=[context],
                                name=plus(prefix, '_layers'),
                                n_steps=nsteps)
    return rval


def param_init_gru_cond(params, dim_in, dim_hid, dim_ctx, prefix = 'att_decoder'):
    W = np.concatenate([norm_weight(dim_in, dim_hid),
                        norm_weight(dim_in, dim_hid)], axis=1)
    params[plus(prefix, 'W')] = W

    U = np.concatenate([norm_weight(dim_hid, dim_hid),
                        norm_weight(dim_hid, dim_hid)], axis=1)
    params[plus(prefix, 'U')] = U

    b = np.zeros((2 * dim_hid,))
    params[plus(prefix, 'b')] = b.astype(theano.config.floatX)

    Wx = norm_weight(dim_in, dim_hid)
    params[plus(prefix, 'Wx')] = Wx
    Ux = norm_weight(dim_hid, dim_hid)
    params[plus(prefix, 'Ux')] = Ux
    params[plus(prefix, 'bx')] = np.zeros((dim_hid,)).astype(theano.config.floatX)

    W_att = norm_weight(dim_ctx, dim_ctx)
    params[plus(prefix, 'W_att')] = W_att

    b_att = np.zeros((dim_ctx,)).astype(theano.config.floatX)
    params[plus(prefix, 'b_att')] = b_att

    W_comb_att = norm_weight(dim_hid, dim_ctx)
    params[plus(prefix, 'W_comb_att')] = W_comb_att

    U_att = norm_weight(dim_ctx, 1)
    params[plus(prefix, 'U_att')] = U_att
    c_att = np.zeros((1, )).astype(theano.config.floatX)
    params[plus(prefix, 'c_att')] = c_att

    Wc = norm_weight(dim_ctx, dim_hid)
    params[plus(prefix, 'Wc')] = Wc
    Uc = norm_weight(dim_hid, dim_hid)
    params[plus(prefix, 'Uc')] = Uc
    params[plus(prefix, 'bc')] = np.zeros((dim_hid,)).astype(theano.config.floatX)

    return params

def gru_cond_layer(tparams, inputs, context, mask=None, context_mask=None, one_step=False, init_memory=None, init_state=None, prefix='att_decoder'):
    assert context, "context must be provided"
    if one_step:
        assert init_state, "previous state must be provided"
    nstep = inputs.shape[0]
    if inputs.ndim == 3:
        n_samples = inputs.shape[1]
    else:
        n_samples = 1

    if mask is None:
        mask = T.alloc(1., inputs.shape[0], 1)

    def _slice(_x, n, m):
        return _x[:, n*m:(n+1)*m]

    dim = tparams[plus(prefix, 'U')].shape[0]

    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)
    assert context.ndim == 3, 'Context must be 3-d'
    pctx_ = T.dot(context, tparams[plus(prefix, 'W_att')]) + tparams[plus(prefix, 'b_att')]

    state_below = T.dot(inputs, tparams[plus(prefix, 'W')]) + tparams[plus(prefix, 'b')]
    state_belowx = T.dot(inputs, tparams[plus(prefix, 'Wx')]) + tparams[plus(prefix, 'bx')]

    def _step(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_):
        preact = T.dot(h_, tparams[plus(prefix, 'U')])
        preact += x_
        r = T.nnet.sigmoid(_slice(preact, 0, dim))
        u = T.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = T.dot(h_, tparams[plus(prefix, 'Ux')])
        preactx *= r
        preactx += xx_

        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        # attention
        pstate = T.dot(h, tparams[plus(prefix, 'W_comb_att')])
        pctx_ = pctx_ + pstate[None, :, :]
        pctx__ = T.tanh(pctx_)
        alpha = T.dot(pctx__, tparams[plus(prefix, 'U_att')]) + tparams[plus(prefix, 'c_att')]
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha_ = T.exp(alpha)
        if context_mask:
            alpha_ = alpha_ * context_mask
        alpha_ = alpha_ / alpha_.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha_[:, :, None]).sum(0)

        h2 =  T.dot(ctx_, tparams[plus(prefix, 'Wc')]) + T.dot(h, tparams[plus(prefix, 'Uc')])
        h2 += tparams[plus(prefix, 'bc')]
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h_

        return h2, ctx_, alpha.T#[M, N]

    seqs = [mask, state_below, state_belowx]
    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context]))
    else:
        rval, updates = theano.scan(_step, sequences=seqs,
                                    outputs_info=[init_state,
                                                  T.alloc(0., n_samples, context.shape[2]),
                                                  T.alloc(0., n_samples, context.shape[0])],
                                    non_sequences=[pctx_, context],
                                    name=plus(prefix, '_layers'),
                                    n_steps = nstep)
    return rval


def param_init_ff(params, dim_in, dim_out, prefix='feed_forward'):
    params[plus(prefix, 'W')] = norm_weight(dim_in, dim_out)
    params[plus(prefix, 'b')] = np.zeros((dim_out,)).astype(theano.config.floatX)

    return params


def ff_layer(tparams, inputs, activ=None, prefix='feed_forward'):
    z = T.dot(inputs, tparams[plus(prefix, 'W')]) + tparams[plus(prefix, 'b')]
    if not activ:
        return z
    if activ == 'tanh':
        return T.tanh(z)
    elif activ == 'softmax':
        return T.nnet.softmax(z)


def dropout_layer(inputs, dropoutRate, trng):
    outputs = inputs * trng.binomial(inputs.shape, p=dropoutRate, n=1, dtype=inputs.dtype)
    return outputs

def param_init_sim(params, dim_left, dim_right, prefix='similarity'):
    params[plus(prefix,'W')] = norm_weight(dim_left, dim_right)

    return params

def sim_layer(tparams, inputs, context, context_mask=None, one_step=False,prefix='similarity'):
    nstep = inputs.shape[0]
    n_samples = inputs.shape[1]
    def _step(x_, a_, ctx_):
        preact = T.dot(ctx_, tparams[plus(prefix, 'W')])
        preact = preact * x_[None, :, :]
        preact = preact.sum(axis=2)
        preact_ = preact.reshape([ctx_.shape[0], ctx_.shape[1]])

        return preact_.T

    seqs = [inputs]
    if one_step:
        rval = _step(*(seqs + [None, context]))
    else:
        rval, updates = theano.scan(_step, sequences=seqs,
                                outputs_info=[T.alloc(0., n_samples, context.shape[0])],
                                non_sequences=[context],
                                name=plus(prefix, '_layers'),
                                n_steps=nstep)
    return rval
