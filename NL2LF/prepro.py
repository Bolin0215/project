import argparse
import json
import os
import nltk
from NL2LF.utils import *
from tqdm import tqdm
import numpy as np

word_counter = {}

def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    source_dir = os.path.join('..', 'data', 'nl2lf')
    target_dir = os.path.join('..', 'data', 'nl2lf')
    glove_dir = os.path.join('..', 'data', 'glove')

    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-tr', '--trainRatio', type=float, default=0.8)
    parser.add_argument('-vr', '--valRatio', type=float, default=0.1)
    parser.add_argument('--pretrain_emb', type=bool, default=False)
    parser.add_argument('--glove_dir', default=glove_dir)
    parser.add_argument('--glove_corpus', default='6B')
    parser.add_argument("--glove_vec_size", default=100, type=int)

    return parser.parse_args()


def prepro(args):
    prepro_each(args)
    if not args.pretrain_emb: return
    word2vec_dict = get_word2vec(args, word_counter)
    save_vec(args, word2vec_dict, 'train_s')


def save(args, data):
    trainset = data[:int(len(data)*args.trainRatio)]
    valset = data[int(len(data)*args.trainRatio):int(len(data)*args.trainRatio)+int(len(data)*args.valRatio)]
    testset = data[int(len(data)*args.trainRatio)+int(len(data)*args.valRatio):]
    data_path = os.path.join(args.target_dir, "data_train.json")
    json.dump(trainset, open(data_path, 'w'))
    data_path = os.path.join(args.target_dir, "data_val.json")
    json.dump(valset, open(data_path, 'w'))
    data_path = os.path.join(args.target_dir, "data_test.json")
    json.dump(testset, open(data_path, 'w'))


def prepro_each(args):
    with open(os.path.join(args.source_dir, 'nfunc_y'), 'r') as f:
        logics = f.readlines()
    with open(os.path.join(args.source_dir, 'nsent_x'), 'r') as f:
        utters = f.readlines()
    import re
    processed = []
    cnt = 0
    for logic, utter in zip(logics, utters):
        if re.search(r'\d{4}-\d{2}-\d{2}', logic) or re.search(r'\d{4}-\d{2}', logic) or  re.search(r'\d{2}-\d{2}-\d{4}', logic):
            continue
        logic = logic.strip()
        for w in logic.split('\t'):
            word_counter.setdefault(w, 0)
            word_counter[w] += 1
        utter = utter.strip().replace(',','').replace('?','')
        # utter = utter.split('-')
        # utter = ' - '.join(utter)
        if utter == '' or logic == '':
            continue
        cnt += 1
        processed.append({'utt':utter, 'logic':logic})

    np.random.shuffle(processed)
    save(args, processed)


def save_vec(args, word2vec_dict, type):
    print('saving word vectors...')
    data_path = os.path.join(args.target_dir, "glove_{}.json".format(type))
    json.dump(word2vec_dict, open(data_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, 'glove.{}.{}d.txt'.format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter),
                                                                        glove_path))
    return word2vec_dict


if __name__ == "__main__":
    main()
