import theano
import theano.tensor as T
from collections import OrderedDict
import numpy as np
from th.nn import param_init_lstm, lstm_layer, param_init_sim
from th.nn import param_init_ff, ff_layer, sim_layer
from th.nn import dropout_layer, param_init_gru_cond, gru_cond_layer
from th.optimizer import adamax, adam
from th.utils import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
theano.config.exception_verbosity='high'
# theano.config.optimizer='fast_compile'

class Model:
    def __init__(self, args, TextData):
        self.args = args
        self.params = None
        self.tparams = None
        self.in_embedded = None
        self.out_embedded = None
        self.in_vocabSize = TextData.in_vocabSize
        self.out_vocabSize = TextData.out_vocabSize

        if self.args.preTrainEmbed:
            self.in_embedded = TextData.word_emb

        self.x = None
        self.x_mask = None
        self.y = None
        self.y_mask = None
        self.dropoutRate = None
        self.predict = None
        self.out = None
        self.cost = None
        self.clip_c = -1.

        self.init_params()
        self.init_tparams()
        self.build_Network()
        self.build_Optimizer()

        self.build_Sampler()
        self.not_save_params = ['Unupdate_out_emb','Update_out_emb','Unupdate_in_emb','Update_in_emb']


    def init_emb(self, params):
        if self.args.preTrainEmbed:
            params['Unupdate_in_emb'] = self.in_embedded.astype(theano.config.floatX)
            length_update = self.in_vocabSize - self.in_embedded.shape[0]
            assert self.in_embedded.shape[1] == self.args.embeddingSize
            print ('Pretrain embedding size is {} new embedding size is {}'.format(self.in_embedded.shape[0], length_update))
            params['Update_in_emb'] = np.zeros((length_update, self.args.embeddingSize), dtype=theano.config.floatX)
        else:
            params['Unupdate_in_emb'] = 0.01*np.random.randn(3, self.args.embeddingSize).astype('float32')
            params['Update_in_emb'] = 0.01*np.random.randn(self.in_vocabSize-3, self.args.embeddingSize).astype('float32')
        params['Unupdate_out_emb'] = 0.01 * np.random.randn(3, self.args.embeddingSize).astype('float32')
        params['Update_out_emb'] = 0.01*np.random.randn(self.out_vocabSize-3, self.args.embeddingSize).astype('float32')
        return params


    def init_params(self):
        self.params = OrderedDict()
        self.params = self.init_emb(self.params)

        self.params = param_init_lstm(self.params, self.args.embeddingSize, self.args.hiddenSize, prefix='lstm_encoder')
        self.params = param_init_lstm(self.params, self.args.embeddingSize, self.args.hiddenSize, prefix='lstm_encoder_r')

        ctxdim = 2 * self.args.hiddenSize

        self.params = param_init_ff(self.params, ctxdim, self.args.hiddenSize, prefix='ff_state')

        self.params = param_init_gru_cond(self.params, self.args.embeddingSize, self.args.hiddenSize, ctxdim, prefix='decoder')

        self.params = param_init_ff(self.params, self.args.hiddenSize, self.out_vocabSize, prefix='logit_lstm')
        if self.args.memorymode:
            self.params = param_init_sim(self.params, self.args.hiddenSize, self.args.hiddenSize, prefix='sim')

    def init_tparams(self):
        self.tparams = OrderedDict()
        for kk, pp in self.params.items():
            self.tparams[kk] = theano.shared(self.params[kk], name=kk)

    def build_Network(self):
        self.x = T.matrix('x', dtype='int64') #[N, M]
        self.x_mask = T.matrix('x_mask', dtype='float32') #[N, M]
        self.y = T.matrix('y', dtype='int64') #[N, M]
        self.y_mask = T.matrix('y_mask', dtype='float32') #[N, M]
        self.dropoutRate = T.fscalar('dropout_rate')
        self.truth_y = T.matrix('truth_y', dtype='int64') #[N, M]
        self.memory_mask = T.ftensor3('memory_mask')

        trng = RandomStreams(200)
        embed_x = T.concatenate([self.tparams['Unupdate_in_emb'], self.tparams['Update_in_emb']], axis=0)
        embed_y = T.concatenate([self.tparams['Unupdate_out_emb'], self.tparams['Update_out_emb']], axis=0)

        embed_x = embed_x[self.x.flatten()]
        embed_x = embed_x.reshape([self.x.shape[0], self.x.shape[1], self.args.embeddingSize])

        if self.args.copymode:
            embed_y = embed_y.dimshuffle(0, 'x', 1)
            embed_y = T.tile(embed_y, (1, self.x.shape[1], 1))
            embed_y = T.concatenate([embed_y, embed_x], axis = 0)
            embed_y = T.transpose(embed_y, (1,0,2))
            embed_y = embed_y[T.arange(self.x.shape[1]), self.y]
        else:
            embed_y = embed_y[self.y.flatten()]
            embed_y = embed_y.reshape([self.y.shape[0], self.y.shape[1], self.args.embeddingSize])

        embed_x = dropout_layer(embed_x, self.dropoutRate, trng)
        embed_y = dropout_layer(embed_y, self.dropoutRate, trng)
        #########################
        # Encoder
        #########################
        Hx_f = lstm_layer(self.tparams, embed_x, self.args.embeddingSize, self.args.hiddenSize, self.x_mask, prefix='lstm_encoder')
        Hx_b = lstm_layer(self.tparams, embed_x[::-1], self.args.embeddingSize, self.args.hiddenSize, self.x_mask[::-1], prefix='lstm_encoder_r')[::-1]
        ctx = T.concatenate([Hx_f, Hx_b], axis=2)

        ctx_mean = (ctx * self.x_mask[:, :, None]).sum(0)/self.x_mask.sum(0)[:, None]
        init_state = ff_layer(self.tparams, ctx_mean, activ='tanh', prefix='ff_state')

        embed_y_shifted = T.zeros_like(embed_y)
        embed_y_shifted = T.set_subtensor(embed_y_shifted[1:], embed_y[:-1])
        embed_y = embed_y_shifted
        #########################
        # Decoder
        #########################

        proj = gru_cond_layer(self.tparams, embed_y, ctx, mask=self.y_mask, context_mask=self.x_mask, one_step=False, init_state=init_state, prefix='decoder')

        proj_h = proj[0] #[N, M, dim]
        ctxs = proj[1] #[N, M, dim_ctx]
        alpha = proj[2] #[N, M, N_en]

        logit = ff_layer(self.tparams, proj_h, prefix='logit_lstm')

        logit_shp = logit.shape
        if self.args.memorymode:
            sim_alpha1 = sim_layer(self.tparams, proj_h, proj_h, self.y_mask, prefix='sim') #[N, M, N]
            sim_alpha = sim_alpha1 * self.memory_mask
            logit_p = T.concatenate([logit, sim_alpha], axis=2)
            logit = T.concatenate(
                [T.exp(logit - logit_p.max(2)[:, :, None]), T.exp(sim_alpha - logit_p.max(2)[:, :, None]) * self.memory_mask],
                axis=2)
            self.out = logit / logit.sum(2, keepdims=True)
            self.out = self.out.reshape([logit.shape[0] * logit.shape[1], logit.shape[2]])
        elif self.args.copymode:
            logit_p = T.concatenate([logit, alpha],axis=2)
            new_mask = self.x_mask.transpose()[None,:,:]
            new_mask = T.tile(new_mask, (alpha.shape[0],1,1))
            logit = T.concatenate([T.exp(logit - logit_p.max(2)[:,:,None]), T.exp(alpha - logit_p.max(2)[:,:,None])* new_mask], axis=2)
            self.out = logit / logit.sum(2, keepdims=True)
            self.out = self.out.reshape([logit.shape[0]*logit.shape[1], logit.shape[2]])
        else:
            self.out = T.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

        if self.args.memorymode:
            self.cost1 = - T.log(self.out[T.arange(self.out.shape[0]), self.truth_y.flatten()] + 1e-10)
        else:
            self.cost1 = - T.log(self.out[T.arange(self.out.shape[0]), self.y.flatten()]  + 1e-10)
        self.cost1 = self.cost1.reshape([self.y.shape[0], self.y.shape[1]])
        self.cost = (self.cost1 * self.y_mask).sum(0).mean()

        self.predict = self.out.argmax(axis=1).reshape([self.y.shape[0], self.y.shape[1]]) * self.y_mask

        if self.args.memorymode:
            self.inputs = [self.x, self.x_mask, self.y, self.y_mask, self.truth_y, self.memory_mask, self.dropoutRate]
            self.inputs_pred = [self.x, self.x_mask, self.y, self.y_mask, self.memory_mask, self.dropoutRate]
            self.f_pred = theano.function(self.inputs_pred, self.predict)
            self.f_debug = theano.function(self.inputs_pred, [self.out, sim_alpha, sim_alpha1])
        else:
            self.inputs = [self.x, self.x_mask, self.y, self.y_mask, self.dropoutRate]
            self.f_pred = theano.function(self.inputs, self.predict)


    def build_Optimizer(self):
        self.grads = T.grad(self.cost, wrt=itemlist(self.tparams))
        if self.clip_c > 0.:
            g2 = 0.
            for g in self.grads:
                g2 += T.sum(g**2)
            new_grads = []
            for g in self.grads:
                new_grads.append(T.switch(g2 > (self.clip_c**2), g/T.sqrt(g2)*self.clip_c, g))
            self.grads = new_grads

        lr = T.scalar(name='lr', dtype=theano.config.floatX)
        self.f_grad_debug = theano.function(self.inputs, self.grads)
        self.f_grad_acc, self.f_update = eval(self.args.optimizer)(lr, self.tparams, self.grads, self.inputs, self.cost, self.args.not_train_params)

    def build_Sampler(self):
        self.x = T.matrix('x', dtype='int64')
        self.x_mask = T.matrix('x_mask', dtype='float32')
        embed_x = T.concatenate([self.tparams['Unupdate_in_emb'], self.tparams['Update_in_emb']], axis=0)
        embed_x = embed_x[self.x.flatten()]
        embed_x = embed_x.reshape([self.x.shape[0], self.x.shape[1], self.args.embeddingSize])
        trng = RandomStreams(200)

        embed_y = T.concatenate([self.tparams['Unupdate_out_emb'], self.tparams['Update_out_emb']], axis=0)

        if self.args.copymode:
            embed_y = T.concatenate([embed_y, embed_x.reshape([self.x.shape[0], self.args.embeddingSize])], axis=0)

        Hx_f = lstm_layer(self.tparams, embed_x, self.args.embeddingSize, self.args.hiddenSize, self.x_mask, prefix='lstm_encoder')
        Hx_b = lstm_layer(self.tparams, embed_x[::-1], self.args.embeddingSize, self.args.hiddenSize, self.x_mask[::-1],
                          prefix='lstm_encoder_r')[::-1]
        ctx = T.concatenate([Hx_f, Hx_b], axis=2)

        ctx_mean = ctx.mean(0)
        init_state = ff_layer(self.tparams, ctx_mean, activ='tanh', prefix='ff_state')

        print ('Building f_init')
        outs = [init_state, ctx]
        self.f_init = theano.function([self.x, self.x_mask], outs, name='f_init')
        print ('Done')

        self.y = T.vector('y_sampler', dtype='int64')#[M]
        init_state = T.matrix('init_state', dtype='float32')#[M, dim]
        self.memory = T.ftensor3('memory')
        self.memory_mask = T.matrix('memory_mask', dtype='float32')
        emb = T.switch(self.y[:,None] < 0,
                       T.alloc(0., 1, self.tparams['Update_out_emb'].shape[1]),
                       embed_y[self.y])
        emb = emb.reshape([self.y.shape[0], self.args.embeddingSize])

        proj = gru_cond_layer(self.tparams, emb, ctx, mask=None, context_mask=self.x_mask, one_step=True, init_state=init_state, prefix='decoder')
        next_state = proj[0] #[M, dim]
        ctxs = proj[1] #[M, dim_ctx]
        alpha = proj[2] #[M, N_en]

        logit = ff_layer(self.tparams, next_state, prefix='logit_lstm')#[M, vocab_size]


        if self.args.copymode:
            logit_p = T.concatenate([logit, alpha], axis=1)
            logit = T.concatenate([T.exp(logit - logit_p.max(1)[:, None]),
                                         T.exp(alpha - logit_p.max(1)[:, None]) * self.x_mask.transpose()], axis=1)
            next_probs = logit / logit.max(1, keepdims=True)
        elif self.args.memorymode:
            memory = self.memory #[N, M, dim]
            sim_alpha = sim_layer(self.tparams, next_state, memory, self.y_mask, one_step=True, prefix='sim')  # [M, N]
            logit_p = T.concatenate([logit, sim_alpha], axis=1)
            logit = T.concatenate(
                [T.exp(logit - logit_p.max(1)[:, None]),
                 T.exp(sim_alpha - logit_p.max(1)[:, None])* self.memory_mask],
                axis=1)
            next_probs = logit / logit.sum(1, keepdims=True)
        else:
            next_probs = T.nnet.softmax(logit)

        next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        print ('Building f_next')
        if self.args.copymode:
            inps = [self.x, self.x_mask, self.y, ctx, init_state]
        elif self.args.memorymode:
            inps = [self.x_mask, self.y, ctx, self.memory, self.memory_mask, init_state]
        else:
            inps = [self.x_mask, self.y, ctx, init_state]
        outs = [next_probs, next_sample, next_state]
        self.f_next = theano.function(inps, outs, name='f_next')
        print ('Done')

    def save_params(self):
        new_params = OrderedDict()
        for kk, vv in self.tparams.items():
            if kk not in self.not_save_params:
                new_params[kk] = vv.get_value()
        return new_params

