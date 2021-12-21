import copy
import pandas as pd
import re

def construct_input_DSA():
    con2id = {}
    id2con = {}
    idx = 0
    node_set = set()

    file = open("dataset/MOOC/DSA/CoreConcepts_DSA", "r")
    for line in file.readlines():
        concepts = line.split("::;")
        id2con[idx] = []
        for concept in concepts:
            concept = concept.split('\n')[0]
            con2id[concept] = idx
            id2con[idx].extend([concept])
            node_set.add(idx)
        idx += 1

    pre_dict = {}
    pre_dict_reverse = {}
    pre_list = []
    for idx in id2con.keys():
        pre_dict_reverse[idx] = set()
        pre_dict[idx] = set()
    file = open("dataset/MOOC/DSA/DSA_LabeledFile", "r")
    for line in file.readlines():
        words = line.strip().split('\t\t')
        A = words[0]
        B = words[1]
        if (words[2] == '1-'):
            pre_dict[con2id[B]].add(con2id[A])
            pre_dict_reverse[con2id[A]].add(con2id[B])
            pre_list.extend([[con2id[B], con2id[A]]])
        elif (words[2] == '-1'):
            pre_dict[con2id[A]].add(con2id[B])
            pre_dict_reverse[con2id[B]].add(con2id[A])
            pre_list.extend([[con2id[A], con2id[B]]])

    file = open("dataset/MOOC/DSA/W-DSA_LabeledFile", "r")
    for line in file.readlines():
        words = line.strip().split('\t\t')
        A = words[0]
        B = words[1]
        if (words[2] == '1-'):
            pre_dict[con2id[B]].add(con2id[A])
            pre_dict_reverse[con2id[A]].add(con2id[B])
            pre_list.extend([[con2id[B], con2id[A]]])
        elif (words[2] == '-1'):
            pre_dict[con2id[A]].add(con2id[B])
            pre_dict_reverse[con2id[B]].add(con2id[A])
            pre_list.extend([[con2id[A], con2id[B]]])

    pre_dict_ = copy.deepcopy(pre_dict)
    for node_1 in pre_dict_.keys():
        for node_2 in pre_dict_[node_1]:
            for node_3 in pre_dict_[node_2]:
                pre_dict[node_1].add(node_3)
                pre_dict_reverse[node_3].add(node_1)
                pre_list.extend([[node_1, node_3]])
    return pre_list, pre_dict, pre_dict_reverse, con2id, id2con, node_set


def construct_input_ML():
    con2id = {}
    id2con = {}
    idx = 0
    node_set = set()

    file = open("dataset/MOOC/ML/CoreConcepts_ML", "r")
    for line in file.readlines():
        concepts = line.split("::;")
        id2con[idx] = []
        for concept in concepts:
            concept = concept.split('\n')[0]
            con2id[concept] = idx
            id2con[idx].extend([concept])
            node_set.add(idx)
        idx += 1

    pre_dict = {}
    pre_dict_reverse = {}
    pre_list = []
    for idx in id2con.keys():
        pre_dict_reverse[idx] = set()
        pre_dict[idx] = set()
    file = open("dataset/MOOC/ML/ML_LabeledFile", "r")
    for line in file.readlines():
        words = line.strip().split('\t\t')
        A = words[0]
        B = words[1]
        if (words[2] == '1-'):
            pre_dict[con2id[B]].add(con2id[A])
            pre_dict_reverse[con2id[A]].add(con2id[B])
            pre_list.extend([[con2id[B], con2id[A]]])
        elif (words[2] == '-1'):
            pre_dict[con2id[A]].add(con2id[B])
            pre_dict_reverse[con2id[B]].add(con2id[A])
            pre_list.extend([[con2id[A], con2id[B]]])

    file = open("dataset/MOOC/ML/W-ML_LabeledFile", "r")
    for line in file.readlines():
        words = line.strip().split('\t\t')
        A = words[0]
        B = words[1]

        if (words[2] == '1-'):
            pre_dict[con2id[B]].add(con2id[A])
            pre_dict_reverse[con2id[A]].add(con2id[B])
            pre_list.extend([[con2id[B], con2id[A]]])

        elif (words[2] == '-1'):
            pre_dict[con2id[A]].add(con2id[B])
            pre_dict_reverse[con2id[B]].add(con2id[A])
            pre_list.extend([[con2id[A], con2id[B]]])

    pre_dict_ = copy.deepcopy(pre_dict)
    for node_1 in pre_dict_.keys():
        for node_2 in pre_dict_[node_1]:
            for node_3 in pre_dict_[node_2]:
                pre_dict[node_1].add(node_3)
                pre_dict_reverse[node_3].add(node_1)
                pre_list.extend([[node_1, node_3]])

    return pre_list, pre_dict, pre_dict_reverse, con2id, id2con, node_set


def construct_input_NPTEL():
    prerequisite_data = pd.read_csv(
        "dataset/PREREQ-IAAI-19-masterdatasets/NPTEL MOOC Dataset/cs_preq.csv",
        header=None)

    node_set = set()
    con2id = {}
    idx = 0
    pre_dict = {}
    pre_dict_reverse = {}
    pre_list = []
    for i in range(len(prerequisite_data)):
        if (prerequisite_data[0][i] not in con2id.keys()):
            con2id[prerequisite_data[0][i]] = idx
            idx += 1
        if (prerequisite_data[1][i] not in con2id.keys()):
            con2id[prerequisite_data[1][i]] = idx
            idx += 1
        if (con2id[prerequisite_data[0][i]] not in pre_dict.keys()):
            pre_dict[con2id[prerequisite_data[0][i]]] = set()
        pre_dict[con2id[prerequisite_data[0][i]]].add(con2id[prerequisite_data[1][i]])
        if (con2id[prerequisite_data[1][i]] not in pre_dict_reverse.keys()):
            pre_dict_reverse[con2id[prerequisite_data[1][i]]] = set()
        pre_dict_reverse[con2id[prerequisite_data[1][i]]].add(con2id[prerequisite_data[0][i]])
        pre_list.extend([[con2id[prerequisite_data[0][i]], con2id[prerequisite_data[1][i]]]])
        node_set.add(con2id[prerequisite_data[0][i]])
        node_set.add(con2id[prerequisite_data[1][i]])

    con2id_new = {}
    for concept in con2id.keys():
        con_new = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", " ", concept)
        con_new = ' '.join(filter(lambda x: x, con_new.split(' ')))
        con2id_new[con_new] = con2id[concept]

    id2con = {}
    for con in con2id_new.keys():
        id2con[con2id_new[con]] = con

    pre_dict_ = copy.deepcopy(pre_dict)
    for node_1 in pre_dict_.keys():
        for node_2 in pre_dict_[node_1]:
            for node_3 in pre_dict_[node_2]:
                pre_dict[node_1].add(node_3)
                pre_dict_reverse[node_3].add(node_1)
                pre_list.extend([[node_1, node_3]])

    return pre_list, pre_dict, pre_dict_reverse, con2id_new, id2con, node_set

def construct_input_University():
    prerequisite_data = pd.read_csv(
        "dataset/PREREQ-IAAI-19-master/datasets/University Course Dataset/cs_preqs.csv",
        header=None)

    node_set = set()
    con2id = {}
    idx = 0
    pre_dict = {}
    pre_dict_reverse = {}
    pre_list = []
    for i in range(len(prerequisite_data)):
        if (prerequisite_data[0][i] not in con2id.keys()):
            con2id[prerequisite_data[0][i]] = idx
            idx += 1
        if (prerequisite_data[1][i] not in con2id.keys()):
            con2id[prerequisite_data[1][i]] = idx
            idx += 1
        if (con2id[prerequisite_data[0][i]] not in pre_dict.keys()):
            pre_dict[con2id[prerequisite_data[0][i]]] = set()
        pre_dict[con2id[prerequisite_data[0][i]]].add(con2id[prerequisite_data[1][i]])
        if (con2id[prerequisite_data[1][i]] not in pre_dict_reverse.keys()):
            pre_dict_reverse[con2id[prerequisite_data[1][i]]] = set()
        pre_dict_reverse[con2id[prerequisite_data[1][i]]].add(con2id[prerequisite_data[0][i]])
        pre_list.extend([[con2id[prerequisite_data[0][i]], con2id[prerequisite_data[1][i]]]])
        node_set.add(con2id[prerequisite_data[0][i]])
        node_set.add(con2id[prerequisite_data[1][i]])

    con2id_new = {}
    for concept in con2id.keys():
        con_new = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", " ", concept)
        con_new = ' '.join(filter(lambda x: x, con_new.split(' ')))
        con2id_new[con_new] = con2id[concept]

    id2con = {}
    for con in con2id_new.keys():
        id2con[con2id_new[con]] = con

    pre_dict_ = copy.deepcopy(pre_dict)
    for node_1 in pre_dict_.keys():
        if node_1 in pre_dict_.keys():
            for node_2 in pre_dict_[node_1]:
                if node_2 in pre_dict_.keys():
                    for node_3 in pre_dict_[node_2]:
                        pre_dict[node_1].add(node_3)
                        pre_dict_reverse[node_3].add(node_1)
                        pre_list.extend([[node_1, node_3]])

    return pre_list, pre_dict, pre_dict_reverse, con2id_new, id2con, node_set


def construct_input_LectureBank():
    prerequisite_data = pd.read_csv(
        "/data/sunhao/code/ConLearn/dataset/LectureBank-master/prerequisite_annotation.csv", header=None)
    checkin_data = pd.read_csv("/data/sunhao/code/ConLearn/dataset/LectureBank-master/208topics.csv",
                               header=None)

    node_set = set()
    con2id = {}
    id2con = {}
    pre_dict = {}
    pre_dict_reverse = {}
    pre_list = []
    for i in range(len(checkin_data)):
        concept = checkin_data[1][i]
        con_new = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", " ", concept)
        con_new = ' '.join(filter(lambda x: x, con_new.split(' ')))
        con2id[con_new] = checkin_data[0][i] - 2
        node_set.add(con2id[con_new])
        id2con[con2id[con_new]] = con_new

    for i in range(len(prerequisite_data)):
        if (prerequisite_data[2][i] == 0):
            continue
        if (prerequisite_data[0][i] > 209 or prerequisite_data[1][i] > 209):
            continue
        conid1 = prerequisite_data[0][i] - 2
        conid2 = prerequisite_data[1][i] - 2
        if (conid1 not in pre_dict.keys()):
            pre_dict[conid1] = set()
        pre_dict[conid1].add(conid2)
        if (conid2 not in pre_dict_reverse.keys()):
            pre_dict_reverse[conid2] = set()
        pre_dict_reverse[conid2].add(conid1)
        pre_list.extend([[conid1, conid2]])

    pre_dict_ = copy.deepcopy(pre_dict)
    for node_1 in pre_dict_.keys():
        if node_1 in pre_dict_.keys():
            for node_2 in pre_dict_[node_1]:
                if node_2 in pre_dict_.keys():
                    for node_3 in pre_dict_[node_2]:
                        pre_dict[node_1].add(node_3)
                        pre_dict_reverse[node_3].add(node_1)
                        pre_list.extend([[node_1, node_3]])

    return pre_list, pre_dict, pre_dict_reverse, con2id, id2con, node_set

def construct_input(dataset):
    if(dataset == 'DSA'):
        pre_list, pre_dict, pre_dict_reverse, con2id, id2con, node_set = construct_input_DSA()
    elif(dataset == 'ML'):
        pre_list, pre_dict, pre_dict_reverse, con2id, id2con, node_set = construct_input_ML()
    elif (dataset == 'NPTEL'):
        pre_list, pre_dict, pre_dict_reverse, con2id, id2con, node_set = construct_input_NPTEL()
    elif (dataset == 'University'):
        pre_list, pre_dict, pre_dict_reverse, con2id, id2con, node_set = construct_input_University()
    elif (dataset == 'LectureBank'):
        pre_list, pre_dict, pre_dict_reverse, con2id, id2con, node_set = construct_input_LectureBank()

    # shuffle is only used in sparsity analysis
    # random.shuffle(pre_list)
    # pre_list = pre_list[:int(0.8 * len(pre_list))]

    pre_dict_ = dict()
    pre_dict_reverse_ = dict()
    for node in pre_dict.keys():
        pre_dict_[node] = list(pre_dict[node])
    for node in pre_dict_reverse.keys():
        pre_dict_reverse_[node] = list(pre_dict_reverse[node])

    return con2id, id2con, pre_dict_, pre_dict_reverse_, pre_list