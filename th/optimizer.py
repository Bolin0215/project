import theano
import numpy as np
import theano.tensor as T


def adamax(lr, tparams, grads, inp, cost, not_train_params, beta1=0.9, beta2=0.999, e=1e-8):
    '''
    :param inp: list
    '''
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_acc = theano.function(inp, cost, updates=gsup)

    updates = []
    t_prev = theano.shared(np.float32(0.), "adamax_t")
    t = t_prev + 1.
    lr_t = lr / (1. - beta1 ** t)

    for p, g in zip(tparams.values(), gshared):
        if p.name in not_train_params:
            continue
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')

        m_t = beta1 * m + (1. - beta1) * g
        v_t = T.maximum(beta2 * v, T.abs_(g))
        step = lr_t * m_t / (v_t + e)

        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    f_update = theano.function([lr], [], updates=updates)
    return f_grad_acc, f_update

def adam(lr, tparams, grads, inp, cost, not_train_params, beta1=0.9, beta2=0.999, e=1e-8):
    '''
    :param inp: list
    '''
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_acc = theano.function(inp, cost, updates=gsup)

    updates = []
    t_prev = theano.shared(np.float32(0.), "adam_t")
    t = t_prev + 1.
    lr_t = lr * T.sqrt(1. - beta2 ** t )/ (1. - beta1 ** t)

    for p, g in zip(tparams.values(), gshared):
        if p.name in not_train_params:
            continue
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')

        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g**2
        step = lr_t * m_t / (T.sqrt(v_t) + e)

        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    f_update = theano.function([lr], [], updates=updates)
    return f_grad_acc, f_update

