import torch
from earlystop import EarlyStopping
import numpy as np

def construct_graph(con1, con2, device):
    items, n_node, A, alias_inputs = [], [], [], []
    con_1= con1.numpy().tolist()
    con_2 = con2.numpy().tolist()

    node_set = set(con_1) | set(con_2)

    max_n_node = len(node_set)
    node = np.array(list(node_set))
    items.append(node.tolist() + (max_n_node - len(node)) * [0])
    u_A = np.zeros((max_n_node, max_n_node))

    instance_1 = []
    instance_2 = []
    for i in range(len(con_1)):
        instance_1.extend([np.where(node == con_1[i])[0][0]])
        instance_2.extend([np.where(node == con_2[i])[0][0]])
        u = np.where(node == con_1[i])[0][0]
        v = np.where(node == con_2[i])[0][0]
        u_A[u][v] = 1
    u_sum_in = np.sum(u_A, 0)
    u_sum_in[np.where(u_sum_in == 0)] = 1
    u_A_in = np.divide(u_A, u_sum_in)
    u_sum_out = np.sum(u_A, 1)
    u_sum_out[np.where(u_sum_out == 0)] = 1
    u_A_out = np.divide(u_A.transpose(), u_sum_out)
    u_A = np.concatenate([u_A_in, u_A_out]).transpose()
    A.append(u_A)

    return torch.FloatTensor(A)[0].to(device), torch.LongTensor(items)[0].to(device), torch.tensor(instance_1).to(device), torch.tensor(instance_2).to(device)

def evaluation(mode, model, eval_dataloader, device):
    with torch.no_grad():
        model.eval()
        count = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for ii, eval_data in enumerate(eval_dataloader):
            A, items, eval_instance_1, eval_instance_2 = construct_graph(eval_data[0][:, 0], eval_data[0][:, 1], device)
            preds = model(items, A, eval_instance_1, eval_instance_2)
            prediction = list(preds[:, 0].cpu().numpy())
            labels = list(eval_data[1].cpu().numpy())
            for i in range(len(prediction)):
                if (prediction[i] > 0.5 and labels[i] == 1):
                    tp += 1
                elif (prediction[i] < 0.5 and labels[i] == 0):
                    tn += 1
                elif (prediction[i] < 0.5 and labels[i] == 1):
                    fn += 1
                elif (prediction[i] > 0.5 and labels[i] == 0):
                    fp += 1
                count += 1
        try:
            precision = tp / (tp + fp)
        except:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except:
            recall = 0
        try:
            f1_score = 2 * precision * recall / (precision + recall)
        except:
            f1_score = 0
        accuracy = (tp + tn) / count
        print(
            mode + "_Accuracy:{:.3f} ".format(accuracy) + mode + "_Precision:{:.3f} ".format(precision) + mode + "_Recall:{:.3f} ".format(recall) + mode + "_F1 Score:{:.3f} ".format(f1_score))
        return f1_score

def train(args, model, optimizer, criterion, train_dataloader, valid_dataloader, test_dataloader, device):

    early_stopping = EarlyStopping(args.patience, verbose=True, save_path=args.save_path, reverse=False)

    for epoch in range(args.epochs):
        ## train
        model.train()
        loss_sum = 0
        for i, train_data in enumerate(train_dataloader):
            A, items, train_instance_1, train_instance_2 = construct_graph(train_data[0][:,0], train_data[0][:,1], device)
            optimizer.zero_grad()
            labels = train_data[1].to(device)
            outputs = model(items, A, train_instance_1, train_instance_2).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss
        print("Epoch {:04d} | loss {:.5f} ".format(epoch, loss_sum / i))

        ## validate
        f1_score_validate = evaluation("Validation", model, valid_dataloader, device)

        early_stopping(f1_score_validate, model)
        if early_stopping.early_stop:
            print("Early Stopping!")
            break
        print()

    ## test
    print("Final Result:")
    model.load_state_dict(torch.load(args.save_path))
    evaluation("Test", model, test_dataloader, device)