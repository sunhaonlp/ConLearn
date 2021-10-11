import argparse
from model import Main_model
import torch
import numpy as np
import random
from utils import construct_input
from dataset import Prerequisite_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from train import train
from pathlib import Path

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):

    args.save_path = Path(
        args.save_path) / "dataset_{}_usebert_{}_updatebert_{}_hiddensize_{}_dropout_{}_lr_{:.4f}_weightdecay_{:.5f}_head_number_{}_batch_size_{}_step_{}.chkpt".format(
        args.dataset, args.usebert, args.updatebert, args.hidden_size, args.dropout_p, args.lr, args.l2, args.n_head, args.batch_size, args.step)

    args.output_dir = Path(
        args.output_dir) / "dataset_{}_usebert_{}_updatebert_{}_hiddensize_{}_dropout_{}_lr_{:.4f}_weightdecay_{:.5f}_head_number_{}_batch_size_{}_step_{}.chkpt".format(
        args.dataset, args.usebert, args.updatebert, args.hidden_size, args.dropout_p, args.lr, args.l2, args.n_head, args.batch_size, args.step)

    set_seeds(args.seed)
    device = torch.device("cuda:{}".format(args.device))
    torch.cuda.set_device(args.device)

    con2id, id2con, pre_dict, pre_dict_reverse, pre_list = construct_input(args.dataset)

    train_dataset = Prerequisite_dataset(args, "train", con2id, id2con, pre_dict, pre_dict_reverse, pre_list)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        drop_last=False)

    valid_dataset = Prerequisite_dataset(args, "validate", con2id, id2con, pre_dict, pre_dict_reverse, pre_list)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        drop_last=False)

    test_dataset = Prerequisite_dataset(args, "test", con2id, id2con, pre_dict, pre_dict_reverse, pre_list)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        drop_last=False)

    model = Main_model(args, len(id2con)).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2)

    criterion = nn.BCELoss()

    train(args, model, optimizer, criterion, train_dataloader, valid_dataloader, test_dataloader, device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Options for Commonsense Knowledge Base Completion')

    parser.add_argument("--load_model", type=str, default=None, help="Path to model file")
    parser.add_argument("--gpu", type=int, default=2,
                        help="gpu")
    parser.add_argument("--no_cuda", action='store_true', default=False,
                        help="prevents using cuda")
    parser.add_argument("--seed", type=int, default=2021,
                        help="random seed value")
    parser.add_argument("--batch_size", type=int, default=512, help="batch_size")
    parser.add_argument("--bert_size", type=int, default=768, help="bert_size")
    parser.add_argument("--glove_size", type=int, default=300, help="glove_size")
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden_size")
    parser.add_argument("--num_workers", type=int, default=5, help="num_workers")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout_p", type=float, default=0.3, help="dropout_p")
    parser.add_argument("--l2", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--device", type=int, default=0, help="device")
    parser.add_argument("--step", type=int, default=1, help="step")
    parser.add_argument("--epochs", type=int, default=100, help="epoch")
    parser.add_argument("--output_dir", type=str, default='./runs/', help="output_dir")
    parser.add_argument("--save_path", type=str, default='./save_models/', help="save_path")
    parser.add_argument("--dataset", type=str, default='DSA', help="dataset")
    parser.add_argument("--usebert", type=bool, default=True, help="usebert")
    parser.add_argument("--useglove", type=bool, default=False, help="useglove")
    parser.add_argument("--n_head", type=int, default=2, help="n_head")
    parser.add_argument("--updatebert", type=bool, default=True, help="updatebert")
    parser.add_argument("--updateglove", type=bool, default=False, help="updateglove")
    parser.add_argument("--patience", type=int, default=15, help="patience")
    parser.add_argument("--idx", type=int, default=0, help="updatebert")

    args = parser.parse_args()
    print(args)

    try:
        main(args)
    except KeyboardInterrupt:
        print('Interrupted')

