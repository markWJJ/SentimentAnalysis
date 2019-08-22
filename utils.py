from collections import Counter
import random


def file2list(in_file, strip_nl=True):
    with open(in_file, 'r', encoding='U8') as f:
        lines = [line.strip() if strip_nl else line for line in f]
    return lines


def file2dict(in_file, split_char='\t', kv_order='01'):
    lines = file2list(in_file)
    items = [line.split(split_char) for line in lines]
    assert len(kv_order) == 2
    k_idx = int(kv_order[0])
    v_idx = int(kv_order[1])
    return {item[k_idx]: item[v_idx] for item in items}

def list2file(out_file, lines, add_nl=True):
    with open(out_file, 'w', encoding='U8') as f:
        f.writelines([f'{line}\n' if add_nl else f'{line}' for line in lines])


def list2stat_file(in_list, out_file):
    c = Counter(in_list)
    stat = c.most_common()
    with open(out_file, 'w', encoding='U8') as f:
        f.writelines([f'{item}\t{count}\n' for item, count in stat])


def merge_file(file_list, out_file, shuffle=False):
    assert isinstance(file_list, (list, tuple))
    ret_lines = []
    for i, file in enumerate(file_list):
        lines = file2list(file, strip_nl=False)
        print(f'已读取第{i}个文件:{file}\t行数{len(lines)}')
        ret_lines.extend(lines)
    if shuffle:
        random.shuffle(ret_lines)
    list2file(out_file, ret_lines, add_nl=False)


def check_overlap(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    print('原始长度', len(list1), '\t', '去重长度', len(set1))
    print('原始长度', len(list2), '\t', '去重长度', len(set2))

    union = list(set1 & set2)
    print('一样的数量', len(union))
    print()

    print('前者多了', set1 - set2)
    print('后者多了', set2 - set1)


def stat_traindata_label():
    """ 统计训练数据类别分布 """
    corpus_path = 'D:/workspace/mutil_turn_intent_detection/corpus/'
    train_files = [corpus_path + 'train_data/' + 'train_9day_ambiscene.clean.txt',
                   corpus_path + 'train_data/' + 'train_9day_keywordscene.clean.txt',
                   corpus_path + 'train_data/' + 'train_9day_chitchat.clean.txt',
                   corpus_path + 'label_data/' + 'test_per1000/other_test.clean.txt']
    total_labels = []
    for file in train_files:
        lines = file2list(file)
        try:
            for line in lines:
                msg, label = line.rsplit('\t', maxsplit=1)
                if label in ['通用', '闲聊']:
                    label = '通用闲聊'
                total_labels.append(label)
        except:
            print(file)
            print(line)
    list2stat_file(total_labels, 'scene_recognition/traindata_label.stat')

def print_len(files):
    length_list = []
    for file in files:
        with open(file, 'r', encoding='U8') as f:
            length_list.append(len(f.readlines()))
    print(length_list, '总和', sum(length_list))

if __name__ == '__main__':
    stat_traindata_label()