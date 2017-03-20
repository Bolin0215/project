import theano
import numpy as np
import os
from tqdm import tqdm
import argparse
from seq2seq.model import Model
import datetime
from seq2seq.textData import textData

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
        parser.add_argument('--lrate', type=float, default=0.02, help='learning rate')
        parser.add_argument('--embeddingSize', type=int, default=50)
        parser.add_argument('--dropoutRate', type=float, default=1.0)
        parser.add_argument('--batchSize', type=int, default=20)
        parser.add_argument('--hiddenSize', type=int, default=56)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--mode', default='train', help='train/test')
        parser.add_argument('--preTrainEmbed', type=bool, default=False)
        parser.add_argument('--optimizer', default='adam')
        parser.add_argument('--attention', type=bool, default=True)
        parser.add_argument('--copymode', type=bool, default=False)
        parser.add_argument('--memorymode',type=bool,default=False)
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
        fout.write('Hidden Layer units is {}\n'.format(self.args.hiddenSize))
        fout.write('Learning rate is {}\n'.format(self.args.lrate))
        fout.write('Batch size is {}\n'.format(self.args.batchSize))
        fout.write('Dropout rate is {}\n'.format(self.args.dropoutRate))	 
        fout.close()
        for e in range(self.args.epochs):
            trainBatches = self.textData.gen_Batches('train')
            totalTrainLoss = 0.0
            totalTrainAcc = 0.0
            bt = datetime.datetime.now()
            xl, yl, predl = [], [], []
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
                    # logit,out1,out = self.model.f_debug(x, x_mask, y, y_mask, dp)
                    # print (x)
                    # print (x_mask)
                    # print (y)
                    # print (y_mask)
                    # print (x_mask.shape)
                    # print (y_mask.shape)
                    # with open('out/params/' + str(self.globalStep), 'w') as f:
                    #     logit = logit.reshape((logit.shape[0]*logit.shape[1], logit.shape[2]))
                    #     for i in range(out1.shape[0]):
                    #         for j in range(out1.shape[1]):
                    #             f.write(str(out1[i][j]) + ' ')
                    #         f.write('\n---------------\n')
                    #     f.write('\n---------------\n')
                    #     for i in range(out.shape[0]):
                    #         for j in range(out.shape[1]):
                    #             f.write(str(out[i][j])  + ' ')
                    #         f.write('\n------------------\n')
                    #     f.write('\n')
                    # # params = self.model.save_params()
                    # # with open('out/params','a+') as f:
                    # #     f.write(str(params))
                    # # return
                    params = self.model.save_params()
                    with open('out/params/' + str(self.globalStep), 'w') as f:
                        f.write(str(params))
                    return

                totalTrainLoss += loss
                if self.args.memorymode:
                    totalTrainAcc += self.accuracy_score(truth_y, predicts)
                else:
                    totalTrainAcc += self.accuracy_score(y, predicts)

                xl.extend(x.transpose().tolist())
                yl.extend(y.transpose().tolist())
                predl.extend(predicts.transpose().tolist())

                self.model.f_update(self.args.lrate)
                params = self.model.save_params()

            out_dir = os.path.join(self.out_dir, 'train')
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            self.show_result(out_dir, e, xl, yl, predl)
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
                        if self.args.copymode:
                            f.write(self.textData.in_idx2word[x_[int(i-self.textData.out_vocabSize)]] + ' ')
                        else:
                            cnt = 0
                            for j in range(idx):
                                if y_[j] == i:
                                    cnt += 1
                            if cnt == 0:
                                f.write(self.textData.out_idx2word[y_[int(i - self.textData.out_vocabSize)]] + ' ')
                            else:
                                f.write(self.textData.out_idx2word[y_[int(i - self.textData.out_vocabSize)]] + '_' + str(cnt))
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
                            cnt = 0
                            for j in range(idx):
                                if p_[j] == i:
                                    cnt += 1
                            if cnt == 0:
                                f.write(self.textData.out_idx2word[p_[int(i - self.textData.out_vocabSize)]] + ' ')
                            else:
                                f.write(
                                    self.textData.out_idx2word[p_[int(i - self.textData.out_vocabSize)]] + '_' + str(
                                        cnt))
                        continue
                    f.write(self.textData.out_idx2word[i] + ' ')
                f.write('\n')

    def gen_sample(self, x, x_mask):
        sample = []
        ret = self.model.f_init(x[:,None], x_mask[:,None])
        next_state, ctx0 = ret[0], ret[1]
        next_w = -1 * np.ones((1,)).astype('int64')

        for i in range(self.args.maxLengthDeco):
            if self.args.copymode:
                inps = [x[:,None], x_mask[:,None], next_w, ctx0, next_state]
            elif self.args.memorymode:
                inps = [x_mask[:,None], next_w, ctx0, next_state]
            else:
                inps = [x_mask[:,None], next_w, ctx0, next_state]
            ret = self.model.f_next(*inps)
            next_p, next_w, next_state = ret[0], ret[1], ret[2]
            nw = next_p[0].argmax()
            sample.append(nw)
            if nw == 2:
                break

        return sample

