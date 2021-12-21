import torch
from earlystop import EarlyStopping
import numpy as np


def generate_negative_samples(id2con, pre_dict, pre_dict_reverse, train_instance, labels):
    len_ = len(train_instance)

    for i in range(int(0.2 * len_)):
        con_a = train_instance[i][0]
        con_b = train_instance[i][1]

        ## Add reverse pair to the dataset
        train_instance.extend([[con_b, con_a]])
        labels.extend([0])

        ## Add random unrelated pair to the dataset
        con_a_negative_b = list(set(id2con.keys()) - set(pre_dict[con_a]))
        negative_a_con_b = list(set(id2con.keys()) - set(pre_dict_reverse[con_b]))

        train_instance.extend([[con_a, con_a_negative_b[np.random.randint(len(con_a_negative_b))]]])
        train_instance.extend([[negative_a_con_b[np.random.randint(len(negative_a_con_b))], con_b]])
        labels.extend([0])
        labels.extend([0])

    return train_instance, labels

def construct_graph(concept_pairs, device, pre_dict, pre_dict_reverse, id2con):
    items, n_node, A, alias_inputs = [], [], [], []

    positive_instance_graph = []
    positive_instance_label = []

    # 50% of prerequisite pairs are used to construct the graph
    threshold = concept_pairs[0].shape[0] * 0.5
    for i in range(concept_pairs[0].shape[0]):
        if(i < threshold):
            positive_instance_graph.extend([[concept_pairs[0][i].item(), concept_pairs[1][i].item()]])
        else:
            positive_instance_label.extend([[concept_pairs[0][i].item(), concept_pairs[1][i].item()]])

    # Oversample 1.5 times
    positive_instance_label.extend(positive_instance_label[:int(0.5 * len(positive_instance_label))])
    labels = [1 for _ in range(len(positive_instance_label))]

    # Generate negative samples
    train_instance, labels = generate_negative_samples(id2con, pre_dict, pre_dict_reverse, positive_instance_label, labels)

    node_set = set()
    for con_pair in train_instance:
        node_set.add(con_pair[0])
        node_set.add(con_pair[1])
    for con_pair in positive_instance_graph:
        node_set.add(con_pair[0])
        node_set.add(con_pair[1])

    # Prerequisite Graph Construction
    node = np.array(list(node_set))
    items.append(node.tolist())
    u_A = np.zeros(( len(node_set),  len(node_set)))
    for i in range(int(len(positive_instance_graph))):
        u = np.where(node == positive_instance_graph[i][0])[0][0]
        v = np.where(node == positive_instance_graph[i][1])[0][0]
        u_A[u][v] = 1
    u_sum_in = np.sum(u_A, 0)
    u_sum_in[np.where(u_sum_in == 0)] = 1
    u_A_in = np.divide(u_A, u_sum_in)
    u_sum_out = np.sum(u_A, 1)
    u_sum_out[np.where(u_sum_out == 0)] = 1
    u_A_out = np.divide(u_A.transpose(), u_sum_out)
    u_A = np.concatenate([u_A_in, u_A_out]).transpose()
    A.append(u_A)

    instance_1 = []
    instance_2 = []
    for con_pair in train_instance:
        instance_1.extend([np.where(node == con_pair[0])[0][0]])
        instance_2.extend([np.where(node == con_pair[0])[0][0]])

    return torch.FloatTensor(A)[0].to(device), torch.LongTensor(items)[0].to(device), torch.tensor(instance_1).to(device), torch.tensor(instance_2).to(device), torch.tensor(labels, dtype=torch.float).to(device)

def evaluation(mode, model, eval_dataloader, device, pre_dict, pre_dict_reverse, id2con):
    with torch.no_grad():
        model.eval()
        count = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for ii, eval_data in enumerate(eval_dataloader):
            A, items, eval_instance_1, eval_instance_2, labels  = construct_graph(eval_data, device, pre_dict, pre_dict_reverse, id2con)
            preds = model(items, A, eval_instance_1, eval_instance_2)
            prediction = list(preds[:, 0].cpu().numpy())
            labels = list(labels.cpu().numpy())
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

def train(args, model, optimizer, criterion, train_dataloader, valid_dataloader, test_dataloader, device, pre_dict, pre_dict_reverse, id2con):

    early_stopping = EarlyStopping(args.patience, verbose=True, save_path=args.save_path, reverse=False)

    for epoch in range(args.epochs):
        ## train
        model.train()
        loss_sum = 0
        for i, train_data in enumerate(train_dataloader):
            A, items, train_instance_1, train_instance_2, labels = construct_graph(train_data, device, pre_dict, pre_dict_reverse, id2con)
            optimizer.zero_grad()
            outputs = model(items, A, train_instance_1, train_instance_2).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss
        print("Epoch {:04d} | loss {:.5f} ".format(epoch, loss_sum / i))

        ## validate
        f1_score_validate = evaluation("Validation", model, valid_dataloader, device, pre_dict, pre_dict_reverse, id2con)

        early_stopping(f1_score_validate, model)
        if early_stopping.early_stop:
            print("Early Stopping!")
            break
        print()

    ## test
    print("Final Result:")
    model.load_state_dict(torch.load(args.save_path))
    evaluation("Test", model, test_dataloader, device, pre_dict, pre_dict_reverse, id2con)