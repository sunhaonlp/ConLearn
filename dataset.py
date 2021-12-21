import random
class Prerequisite_dataset():
    def __init__(self, args, mode, device, pre_dict):
        if(args.dataset == 'LectureBank'):
            ratio_train = 0.8
            ratio_test = 0.9
        else:
            ratio_train = 0.6
            ratio_test = 0.7

        self.data = []
        if (mode == "train"):
            for con in pre_dict.keys():
                self.data.extend([[con, con_2] for con_2 in pre_dict[con][: int(ratio_train * len(pre_dict[con]))]])

        if (mode == "validate"):
            for con in pre_dict.keys():
                self.data.extend([[con, con_2] for con_2 in pre_dict[con][int(ratio_train * len(pre_dict[con])) : int(ratio_test * len(pre_dict[con]))]])

        elif (mode == "test"):
            for con in pre_dict.keys():
                self.data.extend([[con, con_2] for con_2 in pre_dict[con][int(ratio_test * len(pre_dict[con])):]])

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]