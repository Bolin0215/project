import numpy as np
import re

word_split = re.compile('([{},()])')

def prepare():
    from_path = '../data/raw/utter.txt'
    to_path = '../data/raw/logic.txt'

    raw_path = ['../data/raw/output_olympic_debug.txt',\
                '../data/raw/output_movies_debug.txt']
               # '../data/raw/output_sharkAttacks_debug.txt']

    fw_from = open(from_path, 'w')
    fw_to = open(to_path, 'w')
    for i in range(len(raw_path)):
        with open(raw_path[i], 'r') as f:
            res = f.readlines()
        index = 0

        for line in res:
            line = line.strip()
            if index%4 == 0:
                fw_from.write(line + '\n')
            elif index%4 == 2:
                words = []
                # for space_separated_fragment in line:
                words.extend(word_split.split(line))
                words = [w.strip() for w in words if w.strip() and w.strip() not in '(){},']
                fw_to.write('\t'.join(words) + '\n')
            index += 1
    fw_from.close()
    fw_to.close()

def change():
    before_change = '../data/raw/semantics_SharkAttacks_debug.txt'
    after_change = '../data/raw/semantics_SharkAttacks_new.txt'
    with open(before_change, 'r') as f:
        res = f.readlines()
    index = 0
    new_lines = []
    for line in res:
        if index%5 == 1:
            new_line = line.split(']]')[1].strip() + '\n'
        else:
            new_line = line
        new_lines.append(new_line)
        index += 1
    with open(after_change, 'w') as f:
        for line in new_lines:
            f.write(line)


def split(ratio_train=0.8, ratio_dev=0.1):
    from_path = '../data/raw/utter.txt'
    to_path = '../data/raw/logic.txt'

    from_train_path = '../data/prepro/train_utter.txt'
    from_test_path = '../data/prepro/test_utter.txt'
    from_dev_path = '../data/prepro/dev_utter.txt'

    to_train_path = '../data/prepro/train_logic.txt'
    to_test_path = '../data/prepro/test_logic.txt'
    to_dev_path = '../data/prepro/dev_logic.txt'

    paths = [from_train_path, from_dev_path, from_test_path, to_train_path, to_dev_path, to_test_path]

    with open(from_path, 'r') as f:
        from_lines = f.readlines()
    with open(to_path, 'r') as f:
        to_lines = f.readlines()

    assert len(from_lines) == len(to_lines)

    num_total = len(from_lines)
    print ('dataset total num is {}'.format(num_total))

    np.random.seed(200)
    np.random.shuffle(from_lines)
    np.random.seed(200)
    np.random.shuffle(to_lines)

    num_train = int(num_total * ratio_train)
    num_dev = int(num_total * ratio_dev)

    from_train_set = from_lines[:num_train]
    to_train_set = to_lines[:num_train]

    from_dev_set = from_lines[num_train:num_train+num_dev]
    to_dev_set = to_lines[num_train:num_train+num_dev]

    from_test_set = from_lines[num_train+num_dev:]
    to_test_set = to_lines[num_train+num_dev:]

    sets = [from_train_set, from_dev_set, from_test_set, to_train_set, to_dev_set, to_test_set]
    for i in range(6):
        with open(paths[i], 'w') as f:
            for line in sets[i]:
                f.write(line)



if __name__ == "__main__":
    prepare()
    # split()
    # change()
