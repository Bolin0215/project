import theano
import numpy as np
import os
from tqdm import tqdm
import argparse
from seq2seq.model import Model
import datetime
from seq2seq.textData import textData
import copy

class Tagger:
    def __init__(self):
        self.args = None
        self.textData = None
        self.model = None
        self.globalStep = 0
        self.out_file = None

    @staticmethod
    def parseArgs(args):
        parser = argparse.ArgumentParser()
        dir = os.path.join('.','data','nl2lf')
        parser.add_argument('--emb_dir', default=dir,help="embedding vectors directory")
        parser.add_argument('--data_dir', default=dir,help='data directory')
        parser.add_argument('--lrate', type=float, default=0.002, help='learning rate')
        parser.add_argument('--embeddingSize', type=int, default=50)
        parser.add_argument('--dropoutRate', type=float, default=1.0)
        parser.add_argument('--batchSize', type=int, default=20)
        parser.add_argument('--hiddenSize', type=int, default=100)
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--mode', default='train', help='train/test, now only train')
        parser.add_argument('--preTrainEmbed', type=bool, default=False)
        parser.add_argument('--optimizer', default='adam')
        parser.add_argument('--copymode', type=bool, default=False)
        parser.add_argument('--memorymode',type=bool,default=True)
        parser.add_argument('--maxLengthEnco', type=int, default=50)
        parser.add_argument('--maxLengthDeco', type=int, default=70)
        out_dir = os.path.join('.','out')
        parser.add_argument('--out_dir', default=out_dir)

        return parser.parse_args(args)

    def main(self, args=None):
        print ('Theano version v{}'.format(theano.__version__))
        theano.config.floatX = 'float32'
        self.args = self.parseArgs(args)
        if not os.path.isdir(self.args.out_dir):
            os.mkdir(self.args.out_dir)
        self.out_dir = os.path.join(self.args.out_dir, str(self.args.batchSize) + '_' + str(self.args.embeddingSize) + '_' + \
                                     str(self.args.hiddenSize) + '_' + str(self.args.dropoutRate) + '_' + self.args.optimizer +'_' +\
                                     str(self.args.lrate))
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        self.out_file = os.path.join(self.out_dir, 'out')
        assert not os.path.isfile(self.out_file)

        self.args.not_train_params = ['Unupdate_in_emb', 'Unupdate_out_emb']
        self.textData = textData(self.args)
        self.model = Model(self.args, self.textData)

        if self.args.mode == 'train':
            self.train()

    def train(self):
        print ('Start training...')
        fout = open(self.out_file, 'w')
        fout.write('Embedding vector size is {}\n'.format(self.args.embeddingSize))
        fout.write('Hidden layer units is {}\n'.format(self.args.hiddenSize))
        fout.write('Learning rate is {}\n'.format(self.args.lrate))
        fout.write('Batch size is {}\n'.format(self.args.batchSize))
        fout.write('Dropout rate is {}\n'.format(self.args.dropoutRate))	 
        fout.close()
        for e in range(self.args.epochs):
            trainBatches = self.textData.gen_Batches('train')
            totalTrainLoss = 0.0
            totalTrainAcc = 0.0
            bt = datetime.datetime.now()
            xl, truth_yl, predl = [], [], []
            for nextBatch in tqdm(trainBatches):
                self.globalStep += 1
                x, x_mask, y, y_mask, truth_y, memory_mask = nextBatch.x, nextBatch.x_mask, nextBatch.y, nextBatch.y_mask, \
                                                             nextBatch.truth_y, nextBatch.memory_mask
                dp = self.args.dropoutRate
                if self.args.memorymode:
                    loss = self.model.f_grad_acc(x, x_mask, y, y_mask, truth_y, memory_mask, dp)

                    predicts = self.model.f_pred(x, x_mask, y, y_mask, memory_mask, dp)
                else:
                    loss = self.model.f_grad_acc(x, x_mask, y, y_mask, dp)
                    predicts = self.model.f_pred(x, x_mask, y, y_mask, dp)
                if np.isnan(loss):
                    print ('Loss of batch {} is Nan'.format(self.globalStep))
                    return

                totalTrainLoss += loss
                if self.args.memorymode:
                    totalTrainAcc += self.accuracy_score(truth_y, predicts)
                else:
                    totalTrainAcc += self.accuracy_score(y, predicts)

                xl.extend(x.transpose().tolist())
                truth_yl.extend(truth_y.transpose().tolist())
                predl.extend(predicts.transpose().tolist())
                self.model.f_update(self.args.lrate)

            out_dir = os.path.join(self.out_dir, 'train')
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            self.show_result(out_dir, e, xl, truth_yl, predl)
            et = datetime.datetime.now()

            trainLoss = totalTrainLoss / len(trainBatches)
            trainAcc = totalTrainAcc / len(trainBatches)
            valAcc = self.test(e, tag='val')
            testAcc = self.test(e, tag='test')

            print ('epoch = {}/{}, time = {}'.format(e+1, self.args.epochs, et-bt))
            print ('trainLoss = {:.6f}, trainAcc = {:.3f}, valAcc = {:.3f}, testAcc = {:.3f}'.format(trainLoss, trainAcc, valAcc, testAcc))

            fout = open(self.out_file, 'a')
            fout.write('epoch = {}/{}, time = {}, trainLoss = {:.6f}, trainAcc = {:.3f}, valAcc = {:.3f}, testAcc = {:.3f}\n'.format(e+1, self.args.epochs, et-bt, \
                                                                                                     trainLoss, trainAcc, valAcc, testAcc))
            fout.close()

    def test(self, e, tag):
        batches = self.textData.gen_Batches(tag)
        totalAcc = 0.0
        xl, truth_yl, predl = [], [], []
        for nextBatch in batches:
            x, x_mask, y, y_mask, truth_y, memory_mask = nextBatch.x, nextBatch.x_mask, nextBatch.y, nextBatch.y_mask, \
                                                         nextBatch.truth_y, nextBatch.memory_mask
            predicts = []
            xl.extend(x.transpose().tolist())
            if self.args.memorymode:
                truth_yl.extend(truth_y.transpose().tolist())
            else:
                truth_yl.extend(y.transpose().tolist())
            for i in range(x.shape[1]):
                predict = self.gen_sample(x[:,i], x_mask[:,i])
                predl.append(predict)
                predicts.append(predict)
            if self.args.memorymode:
                totalAcc += self.accuracy_score(truth_y, predicts)
            else:
                totalAcc += self.accuracy_score(y, predicts)
        totalAcc /= len(batches)
        out_dir = os.path.join(self.out_dir, tag)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        self.show_result(out_dir, e, xl, truth_yl, predl)
        return totalAcc

    def accuracy_score(self, y, pred):
        y = y.transpose().tolist()
        if not isinstance(pred, list):
            pred = pred.transpose().tolist()
        acc = 0
        for y_, pred_ in zip(y, pred):
            align = True
            for y__, p__ in zip(y_, pred_):
                if p__ == 2 and y__ == 2:
                    break
                if y__ != p__:
                    align = False
                    break
            if align:
                acc += 1
        return acc / len(y)

    def show_result(self, dir, e, x, y, pred):
        out_file = os.path.join(dir, str(e))
        with open(out_file, 'w') as f:
            for x_, y_, p_ in zip(x, y, pred):
                f.write('X: ')
                for i in x_:
                    if i == 0: break
                    f.write(self.textData.in_idx2word[i] + ' ')
                f.write('\n')
                f.write('Y: ')
                for idx, i in enumerate(y_):
                    if i == 0 or i == 2: break
                    if i >= self.textData.out_vocabSize:
                        # Memory mode:
                        # consider this example: ..... sandbox.medals .... sandbox.medals
                        # the 2nd medals is a reference of the 1st medals
                        # so the index of the 2nd medals is the out_vocabSize + pos(1st medals)
                        if self.args.copymode:
                            f.write(self.textData.in_idx2word[x_[int(i-self.textData.out_vocabSize)]] + ' ')
                        else:
                            tmpi = i - self.textData.out_vocabSize
                            index = int(tmpi)
                            tmpi = y_[int(tmpi)]
                            while tmpi >= self.textData.out_vocabSize:
                                tmpi = tmpi - self.textData.out_vocabSize
                                index = int(tmpi)
                                tmpi = y_[int(tmpi)]
                            cnt = 0
                            for j in range(index):
                                if y_[j] == tmpi: cnt += 1
                            if cnt == 0:
                                f.write(self.textData.out_idx2word[tmpi] + ' ')
                            else:
                                f.write(
                                    self.textData.out_idx2word[tmpi] + '_' + str(
                                        cnt) + ' ')
                        continue
                    if self.args.memorymode and 'sandbox' in self.textData.out_idx2word[i] and len(self.textData.out_idx2word[i].split('.')) == 2:
                        # Memory mode:
                        # consider this example: ..... sandbox.medals .... sandbox.medals_1
                        # the 2nd medals is a new table, so the notation of the 2nd medals is different from the 1st.
                        # but the index of the 2nd medals is in the out_vocabSize
                        cnt = 0
                        for j in range(idx):
                            if y_[j] == i: cnt += 1
                        if cnt == 0:
                            f.write(self.textData.out_idx2word[i] + ' ')
                        else:
                            f.write(
                                    self.textData.out_idx2word[i] + '_' + str(
                                        cnt) + ' ')
                        continue
                    f.write(self.textData.out_idx2word[i] + ' ')
                f.write('\n')
                f.write('P: ')
                for idx, i in enumerate(p_):
                    if i == 0 or i == 2: break
                    if i >= self.textData.out_vocabSize:
                        if self.args.copymode:
                            f.write(self.textData.in_idx2word[x_[int(i-self.textData.out_vocabSize)]] + ' ')
                        else:
                            tmpi = i - self.textData.out_vocabSize
                            index = int(tmpi)
                            tmpi = p_[int(tmpi)]
                            while tmpi >= self.textData.out_vocabSize:
                                tmpi = tmpi - self.textData.out_vocabSize
                                index = int(tmpi)
                                tmpi = p_[int(tmpi)]
                            cnt = 0
                            for j in range(index):
                                if p_[j] == tmpi: cnt += 1
                            if cnt == 0:
                                f.write(self.textData.out_idx2word[tmpi] + ' ')
                            else:
                                f.write(
                                    self.textData.out_idx2word[tmpi] + '_' + str(
                                        cnt) + ' ')
                        continue
                    if self.args.memorymode:
                        cnt = 0
                        for j in range(idx):
                            if p_[j] == i: cnt += 1
                        if cnt == 0 or len(self.textData.out_idx2word[i].split('.')) != 2:
                            f.write(self.textData.out_idx2word[i] + ' ')
                        else:
                            f.write(
                                self.textData.out_idx2word[i] + '_' + str(
                                    cnt) + ' ')
                        continue
                    f.write(self.textData.out_idx2word[i] + ' ')
                f.write('\n')

    def gen_sample(self, x, x_mask):
        sample = []
        ret = self.model.f_init(x[:,None], x_mask[:,None])
        next_state, ctx0 = ret[0], ret[1]
        next_w = -1 * np.ones((1,)).astype('int64')
        memory = np.array([],dtype='float32')
        memory_mask = np.zeros((1,1)).astype('float32')

        for i in range(self.args.maxLengthDeco):
            # print (i)
            if self.args.copymode:
                inps = [x[:,None], x_mask[:,None], next_w, ctx0, next_state]
            elif self.args.memorymode:
                if i == 0:
                    tmpmemory = np.zeros(next_state.shape).astype('float32')
                else:
                    tmpmemory = memory
                tmpmemory = tmpmemory.reshape([tmpmemory.shape[0], 1, tmpmemory.shape[1]])
                inps = [x_mask[:,None], next_w, ctx0, tmpmemory, memory_mask, next_state]
            else:
                inps = [x_mask[:,None], next_w, ctx0, next_state]
            ret = self.model.f_next(*inps)
            next_p, next_w, next_state = ret[0], ret[1], ret[2]
            while next_w[0] >= self.textData.out_vocabSize:
                next_w[0] -= self.textData.out_vocabSize
                next_w[0] = sample[next_w[0]]
            nw = next_p[0].argmax()
            if i == 0:
                memory = next_state
                if nw >= self.textData.out_vocabSize or (nw >= 3 and nw < self.textData.table_count + 3):
                    memory_mask = np.ones((1,1)).astype('float32')
                else:
                    memory_mask = np.zeros((1,1)).astype('float32')
            else:
                memory = np.concatenate([memory, next_state], axis=0)
                if nw >= self.textData.out_vocabSize or (nw >= 3 and nw < self.textData.table_count + 3):
                    memory_mask = np.concatenate([memory_mask, np.ones((1,1)).astype('float32')], axis=1)
                else:
                    memory_mask = np.concatenate([memory_mask, np.zeros((1,1)).astype('float32')], axis=1)

            sample.append(nw)
            if nw == 2:
                break

        return sample

    def beam_sample(self, x, x_mask):
        sample = []
        sample_score = []
        k = 5 #beam size
        live_k = 5
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')
        hyp_states = []

        ret = self.model.f_init(x[:,None], x_mask[:,None])
        next_state, ctx0 = ret[0], ret[1]
        next_w = -1 * np.ones((1,)).astype('int64')
        memory = np.array([],dtype='float32')
        memory_mask = np.zeros((1,1)).astype('float32')
        # print (ctx0)

        for i in range(self.args.maxLengthDeco):
            if i != 0:
                ctx = np.tile(ctx0, [live_k, 1])
                mask = np.tile(x_mask[:,None], [1,live_k])
            else:
                ctx = ctx0
                mask = x_mask[:,None]
            if self.args.copymode:
                inps = [x[:,None], x_mask[:,None], next_w, ctx, next_state]
            elif self.args.memorymode:
                if i == 0:
                    tmpmemory = np.zeros(next_state.shape).astype('float32')
                    tmpmemory = tmpmemory.reshape([tmpmemory.shape[0], 1, tmpmemory.shape[1]])
                else:
                    tmpmemory = memory
                inps = [mask, next_w, ctx, tmpmemory, memory_mask, next_state]
            else:
                inps = [mask, next_w, ctx, next_state]
            ret = self.model.f_next(*inps)
            next_p, next_w, next_state = ret[0], ret[1], ret[2]
            if i == 0:
                memory = tmpmemory
            memory = np.transpose(memory, (1, 0, 2))

            if i == 0:
                cand_scores = - np.log(next_p.flatten())
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k - dead_k)]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size
                word_indices = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]

            else:
                cand_scores = hyp_scores[:, None] - np.log(next_p) #[[p_word1, p_word2, ...],[p_word1, p_word2, ...]...]
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k-dead_k)]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size
                word_indices = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = np.zeros(k - dead_k).astype('float32')
            new_hyp_states = []
            new_memory = []
            new_memory_mask = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                ti = int(ti)
                wi = int(wi)
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                nw = wi
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

                if i == 0:
                    new_memory.append(copy.copy(next_state[ti]))
                    if nw >= self.textData.out_vocabSize or (nw >= 3 and nw < self.textData.table_count + 3):
                        new_memory_mask.append(np.ones((1,)).astype('float32'))
                    else:
                        new_memory_mask.append(np.zeros((1,)).astype('float32'))
                else:
                    tmpmemory = np.concatenate([memory[ti], next_state[ti]])
                    new_memory.append(tmpmemory)
                    if nw >= self.textData.out_vocabSize or (nw >= 3 and nw < self.textData.table_count + 3):
                        tmpmemory_mask = np.concatenate([memory_mask[ti], np.ones((1,)).astype('float32')])
                        new_memory_mask.append(tmpmemory_mask)
                    else:
                        tmpmemory_mask = np.concatenate([memory_mask[ti], np.zeros((1,)).astype('float32')])
                        new_memory_mask.append(tmpmemory_mask)

            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            memory = []
            memory_mask = []

            for idx in range(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 2:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    memory.append(new_memory[idx])
                    memory_mask.append(new_memory_mask[idx])

            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = np.array([w[-1] for w in hyp_samples])
            next_state = np.array(hyp_states)
            memory = np.array(memory)
            if i == 0:
                memory = memory.reshape([memory.shape[0], 1, memory.shape[1]])
            memory = np.transpose(memory, (1,0,2))
            memory_mask = np.array(memory_mask)

        if live_k > 0:
            for idx in range(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
        sample_score = np.array(sample_score)
        ss = sample[sample_score.argmin()]
        return ss

