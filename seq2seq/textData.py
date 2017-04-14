import os
import json
import numpy as np
from collections import defaultdict
import re

class pair:
    def __init__(self):
        self.x = None
        self.y = None
        self.truth_y = None

class Batch:
    def __init__(self):
        self.x = None
        self.y = None
        self.truth_y = None
        self.x_mask = None
        self.y_mask = None
        self.memory_mask = None

class textData:
    def __init__(self, args):
        self.args = args
        self.in_word2idx = {}
        self.in_idx2word = {}

        self.out_word2idx = {}
        self.out_idx2word = {}
        self.out_wordcount = defaultdict(int)

        self.word_emb = []

        if self.args.preTrainEmbed:
            self.gen_pretrain_word_vec()
        self.untrainableCnt = len(self.word_emb)
        print ('{} words embedding are untrainable'.format(self.untrainableCnt))
        self.in_unk_word = 'UNK' #actually UNK is not used here.
        self.in_pad_word = 'PAD'
        self.in_eos_word = 'EOS'

        self.out_unk_word = 'UNK'
        self.out_pad_word = 'PAD'
        self.out_eos_word = 'EOS'

        self.in_word2idx[self.in_pad_word] = self.untrainableCnt
        self.in_idx2word[self.untrainableCnt] = self.in_pad_word
        self.in_word2idx[self.in_unk_word] = self.untrainableCnt + 1
        self.in_idx2word[self.untrainableCnt + 1] = self.in_unk_word
        self.in_word2idx[self.in_eos_word] = self.untrainableCnt + 2
        self.in_idx2word[self.untrainableCnt + 2] = self.in_eos_word

        self.out_word2idx[self.out_pad_word] = 0
        self.out_idx2word[0] = self.out_pad_word
        self.out_word2idx[self.out_unk_word] = 1
        self.out_idx2word[1] = self.out_unk_word
        self.out_word2idx[self.out_eos_word] = 2
        self.out_idx2word[2] = self.out_eos_word

        self.total_x_idx = self.untrainableCnt + 3
        self.total_y_idx = 3
        self.table_count = 0

        if self.args.memorymode:
            self.load_table()
            self.table_count = self.total_y_idx - 3
            self.load_word()

            self.in_vocabSize = len(self.in_word2idx)
            self.out_vocabSize = len(self.out_word2idx)
            assert self.total_x_idx == self.in_vocabSize
            assert self.total_y_idx == self.out_vocabSize
        self.read_all_data()

        self.in_vocabSize = len(self.in_word2idx)
        self.out_vocabSize = len(self.out_word2idx)
        assert self.total_x_idx == self.in_vocabSize
        assert self.total_y_idx == self.out_vocabSize


        print ('X: total {} words\n'.format(self.in_vocabSize))
        print ('Y: total {} words\n'.format(self.out_vocabSize))

    def gen_pretrain_word_vec(self):
        vec_path = os.path.join(self.args.emb_dir, "glove_train.json")
        with open(vec_path, 'r') as f:
            vecs = json.load(f)
        vectors = []
        for idx, word in enumerate(vecs):
            self.in_word2idx[word] = idx
            self.in_idx2word[idx] = word
            vectors.append(vecs[word])
        self.word_emb = np.array(vectors).astype(np.float32)

    def load_table(self):
        data_set = ['train', 'test', 'val']
        for data_type in data_set:
            data_path = os.path.join(self.args.data_dir, "data_{}.json".format(data_type))
            with open(data_path, 'r') as f:
                datas = json.load(f)
            for data in datas:
                sent_l = data['logic']
                for w in sent_l.split('\t'):
                    raw_w = w.replace('\'', '').replace('#', '')
                    w = w.lower()
                    if w.startswith('sandbox') and len(w.split('.')) == 2:
                        if re.search('^\d+$', w.split('_')[-1]):
                            w = '_'.join(w.split('_')[:-1])
                        else:
                            w = w
                        if w not in self.out_word2idx:
                            self.out_word2idx[w] = self.total_y_idx
                            self.out_idx2word[self.total_y_idx] = w
                            self.total_y_idx += 1

    def load_word(self):
        data_set = ['train', 'test', 'val']
        for data_type in data_set:
            data_path = os.path.join(self.args.data_dir, "data_{}.json".format(data_type))
            with open(data_path, 'r') as f:
                datas = json.load(f)
            for data in datas:
                sent = data['utt']
                for w in sent.split('\t'):
                    w = w.lower()
                    if w not in self.in_word2idx:
                        self.in_word2idx[w] = self.total_x_idx
                        self.in_idx2word[self.total_x_idx] = w
                        self.total_x_idx += 1
                sent_l = data['logic']
                for w in sent_l.split('\t'):
                    raw_w = w.replace('\'','').replace('#','')
                    w = w.lower()
                    if '.' in w:
                        self.out_wordcount[w] += 1
                    if self.args.memorymode and len(w.split('.'))==2 and 'sandbox' in w:
                        if re.search('^\d+$', w.split('_')[-1]):
                            w = '_'.join(w.split('_')[:-1])
                        else:
                            w = w
                    if w not in self.out_word2idx:
                        if self.args.copymode and w in sent.lower().split('\t') and w != 'show':
                            continue
                        else:
                            self.out_word2idx[w] = self.total_y_idx
                            self.out_idx2word[self.total_y_idx] = w
                            self.total_y_idx += 1

    def _read_data(self, data_type):
        data_path = os.path.join(self.args.data_dir, "data_{}.json".format(data_type))
        with open(data_path, 'r') as f:
            datas = json.load(f)

        num_examples = len(datas)

        new_datas = []
        for num, data in enumerate(datas):
            npair = pair()
            sent = data["utt"]
            x_ids = []
            for w in sent.split('\t'):
                w = w.lower()
                if w in self.in_word2idx:
                    x_ids.append(self.in_word2idx[w])
                else:
                    assert not self.args.copymode
                    self.in_word2idx[w] = self.total_x_idx
                    self.in_idx2word[self.total_x_idx] = w
                    x_ids.append(self.total_x_idx)
                    self.total_x_idx += 1
            npair.x = x_ids
            if len(x_ids) > self.args.maxLengthEnco:
                continue
            y_ids = [] # fake index of y, is used to get embedding
            # in current version, used table and new table have same token embedding
            truth_y_ids = [] # truth index of y
            sent_l = data["logic"]
            for id, w in enumerate(sent_l.split('\t')):
                raw_w = w.replace('\'','').replace('#','')
                w = w.lower()
                if w in self.out_word2idx:
                    if self.args.memorymode and len(w.split('.')) == 2 and 'sandbox' in w:

                        found = False
                        for j in range(id):
                            if sent_l.split('\t')[j].lower() == w:
                                y_ids.append(y_ids[j])
                                for ji in range(id):
                                    jk = id - ji - 1
                                    if sent_l.split('\t')[jk].lower() == w:
                                        truth_y_ids.append(self.out_vocabSize + jk)
                                        # the table is used before
                                        break
                                found = True
                                break
                        if not found:
                            # the table is a new table
                            y_ids.append(self.out_word2idx[w])
                            truth_y_ids.append(self.out_word2idx[w])
                        continue
                    y_ids.append(self.out_word2idx[w])
                    truth_y_ids.append(self.out_word2idx[w])
                elif self.args.copymode and w != 'show' and raw_w in sent.split('\t'):
                    y_ids.append(self.total_y_idx + sent.split('\t').index(raw_w))
                else:
                    assert not self.args.copymode
                    if self.args.memorymode and len(w.split('.')) == 2 and 'sandbox' in w:

                        found = False
                        for j in range(id):
                            if sent_l.split('\t')[j].lower() == w:
                                y_ids.append(y_ids[j])
                                for ji in range(id):
                                    jk = id - ji - 1
                                    if sent_l.split('\t')[jk].lower() == w:
                                        truth_y_ids.append(self.out_vocabSize + jk)
                                        # the table is used before
                                        break
                                found = True
                                break
                        if not found:
                            # the table is a new table
                            if self.args.memorymode and len(w.split('.')) == 2 and 'sandbox' in w:
                                if re.search('^\d+$', w.split('_')[-1]):
                                    w = '_'.join(w.split('_')[:-1])
                                else:
                                    w = w
                            y_ids.append(self.out_word2idx[w])
                            truth_y_ids.append(self.out_word2idx[w])
                    else:
                        assert not self.args.memorymode
                        # not a table
                        self.out_word2idx[w] = self.total_y_idx
                        self.out_idx2word[self.total_y_idx] = w
                        y_ids.append(self.total_y_idx)
                        truth_y_ids.append(self.total_y_idx)
                        self.total_y_idx += 1
            y_ids.append(2)
            truth_y_ids.append(2)
            assert len(y_ids) == len(truth_y_ids)
            npair.y = y_ids
            npair.truth_y = truth_y_ids
            if len(y_ids) > self.args.maxLengthDeco:
                continue
            new_datas.append(npair)

        np.random.shuffle(new_datas)
        print('{} total {} examples... '.format(data_type, len(new_datas)))
        return new_datas

    def read_all_data(self):
        self.train_data = self._read_data('train')
        self.val_data = self._read_data('val')
        self.test_data = self._read_data('test')

    def gen_Batches(self, data_type):
        batches_sample = []
        if data_type == 'train':
            samples = self.train_data
        elif data_type == 'val':
            samples = self.val_data
        else:
            samples = self.test_data
        numSamples = len(samples)
        np.random.shuffle(samples)
        for i in range(0, numSamples, self.args.batchSize):
            batch_sample = samples[i: min(i + self.args.batchSize, numSamples)]
            max_x_num, max_y_num = -1, -1
            for pair in batch_sample:
                max_x_num = len(pair.x) if max_x_num < len(pair.x) else max_x_num
                max_y_num = len(pair.y) if max_y_num < len(pair.y) else max_y_num

            new_x = np.zeros((len(batch_sample), max_x_num)).astype(np.int64)
            new_x_mask = np.zeros((len(batch_sample), max_x_num)).astype(np.float32)
            new_y = np.zeros((len(batch_sample), max_y_num)).astype(np.int64)
            new_y_mask = np.zeros((len(batch_sample), max_y_num)).astype(np.float32)
            new_truth_y = np.zeros((len(batch_sample), max_y_num)).astype(np.int64)
            new_memory_mask = np.zeros((max_y_num, len(batch_sample), max_y_num)).astype(np.float32)

            for idx, pair in enumerate(batch_sample):
                new_x[idx][:len(pair.x)] = pair.x
                new_x_mask[idx][:len(pair.x)] = 1.
                new_y[idx][:len(pair.y)] = pair.y
                new_y_mask[idx][:len(pair.y)] = 1.
                new_truth_y[idx][:len(pair.truth_y)] = pair.truth_y
            for idx in range(max_y_num):
                for jdx in range(len(batch_sample)):
                    for kdx in range(idx):
                        if new_truth_y[jdx][kdx] >= self.out_vocabSize or \
                            (new_truth_y[jdx][kdx] < self.table_count + 3 and new_truth_y[jdx][kdx] >= 3):
                            # the token is a table
                            new_memory_mask[idx][jdx][kdx] = 1.

            new_batch = Batch()
            new_batch.x = new_x.transpose()
            new_batch.x_mask = new_x_mask.transpose()
            new_batch.y = new_y.transpose()
            new_batch.y_mask = new_y_mask.transpose()
            new_batch.truth_y = new_truth_y.transpose()
            new_batch.memory_mask = new_memory_mask
            batches_sample.append(new_batch)

        return batches_sample
