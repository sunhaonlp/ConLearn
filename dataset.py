import torch
import numpy as np

class Prerequisite_dataset():
    def __init__(self, args, mode, device, id2con, pre_dict, pre_dict_reverse, pre_list):
        if(args.dataset == 'LectureBank'):
            ratio_train = 0.8
            ratio_test = 0.9
        else:
            ratio_train = 0.6
            ratio_test = 0.7

        self.device = device
        if (mode == "train"):
            self.data_ = [[pre_list[i][0], pre_list[i][1]] for i in range(int(ratio_train * len(pre_list)) - 1)]

            self.data = [[pre_list[i][0], pre_list[i][1]] for i in range(int(ratio_train * len(pre_list)) - 1)]
            self.data.extend([[pre_pair[0], pre_pair[1]] for pre_pair in self.data[:int(0.5 * len(self.data))]])
            self.labels = [1 for _ in range(int(ratio_train * len(pre_list)) - 1)]
            self.labels.extend([1 for _ in range(int(len(self.labels)*0.5))])

        if (mode == "validate"):
            self.data_ = [[pre_list[i][0], pre_list[i][1]] for i in range(int(ratio_train * len(pre_list)), int(ratio_test * len(pre_list)) - 1)]

            self.data = [[pre_list[i][0], pre_list[i][1]] for i in range(int(ratio_train * len(pre_list)), int(ratio_test * len(pre_list)) - 1)]
            self.data.extend([[pre_pair[0], pre_pair[1]] for pre_pair in self.data[:int(0.5 * len(self.data))]])
            self.labels = [1 for _ in range(int(ratio_train * len(pre_list)), int(ratio_test * len(pre_list)) - 1)]
            self.labels.extend([1 for _ in range(int(len(self.labels)*0.5))])

        elif (mode == "test"):
            self.data_ = [[pre_list[i][0], pre_list[i][1]] for i in range(int(ratio_test * len(pre_list)), len(pre_list))]

            self.data = [[pre_list[i][0], pre_list[i][1]] for i in range(int(ratio_test * len(pre_list)), len(pre_list))]
            self.data.extend([[pre_pair[0], pre_pair[1]] for pre_pair in self.data[:int(0.5 * len(self.data))]])
            self.labels = [1 for _ in range(int(ratio_test * len(pre_list)), len(pre_list))]
            self.labels.extend([1 for _ in range(int(len(self.labels)*0.5))])

        self.generate_negative_samples(id2con, pre_dict, pre_dict_reverse)

    def generate_negative_samples(self, id2con, pre_dict, pre_dict_reverse):

        for i in range(len(self.data_)):
            con_a = self.data_[i][0]
            con_b = self.data_[i][1]

            ## Add reverse pair to the dataset
            self.data.extend([[con_b, con_a]])
            self.labels.extend([0])

            ## Add random unrelated pair to the dataset
            con_a_negative_b = list(set(id2con.keys()) - set(pre_dict[con_a]))
            negative_a_con_b = list(set(id2con.keys()) - set(pre_dict_reverse[con_b]))

            self.data.extend([[con_a, con_a_negative_b[np.random.randint(len(con_a_negative_b))]]])
            self.data.extend([[negative_a_con_b[np.random.randint(len(negative_a_con_b))], con_b]])
            self.labels.extend([0])
            self.labels.extend([0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)