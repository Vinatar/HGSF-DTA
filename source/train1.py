import csv
import os
import json
from datetime import datetime

import torch
import numpy as np
from collections import OrderedDict

from model import BGNN, Predictor
from metrics import model_evaluate
from GraphInput import getAffinityGraph, getDrugMolecularGraph, getTargetMolecularGraph
from myutils import argparser, DTADataset, GraphDataset, collate, getLinkEmbeddings, predicting, read_data, train

from preprocessing import process_target_data, process_drug_data


# 从文件中加载药物-靶点亲和性数据集
# 根据指定的折叠（fold）创建训练集和测试集的数据。
# 返回训练集、测试集以及药物-靶点亲和性图。
def create_dataset_for_train_test(affinity, dataset, fold, weighted, drug_aff_k, target_aff_k):
    # load dataset
    dataset_path = 'data/' + dataset + '/'

    # 处理药物smiles和靶点序列嵌入
    drug_data_path = dataset_path + 'drugs.txt'
    drug_list = process_drug_data(drug_data_path)
    target_data_path = dataset_path + 'targets.txt'
    target_list = process_target_data(target_data_path)

    train_fold_origin = json.load(open(dataset_path + 'S1_train_set.txt'))
    train_folds = []
    print("len(train_fold_origin)=====", str(len(train_fold_origin)))
    for i in range(len(train_fold_origin)):
        if i != fold:
            train_folds += train_fold_origin[i]
    # test_fold 是测试集的索引，如果 fold 等于 -100，则使用
    test_fold = json.load(open(dataset_path + 'S1_test_set.txt')) if fold == -100 else train_fold_origin[fold]

    # rows, cols 通过 np.where 获取亲和性矩阵中非 NaN（存在亲和性值）的行和列的索引。
    rows, cols = np.where(np.isnan(affinity) == False)

    # train_rows, train_cols 是训练集的行和列索引，仅包含训练折叠对应的索引。
    # 切片操作 当前已知亲和力矩阵68*442=30056，得到非空行列索引rows, cols,再从中rows[], cols[]选出满足train_folds（里面存储的是训练集中药靶对的序号）的行列索引
    # rows[train_folds]：这是在数组 rows 中选择满足条件 train_folds 的行。结果是一个包含满足条件的行的新数组，称为 train_rows。
    # cols[train_folds]：这是在数组 cols 中选择满足条件 train_folds 的列。结果是一个包含满足条件的列的新数组，称为 train_cols。
    train_rows, train_cols = rows[train_folds], cols[train_folds]
    # print("rows[1483]="+str(rows[1483])+" cols[1483]="+str(cols[1483]))

    train_Y = affinity[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, drug_smiles=drug_list, target_sequences=target_list, y=train_Y)

    test_rows, test_cols = rows[test_fold], cols[test_fold]
    test_Y = affinity[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, drug_smiles=drug_list, target_sequences=target_list, y=test_Y)

    # adj 是一个与亲和性矩阵相同形状的全零矩阵
    adj = np.zeros_like(affinity)
    adj[train_rows, train_cols] = train_Y
    affinity_graph = getAffinityGraph(dataset, adj, weighted, drug_aff_k, target_aff_k)

    return train_dataset, test_dataset, affinity_graph


def train_test():
    # 通过命令行参数（argparser）获取配置信息。
    FLAGS = argparser()

    dataset = FLAGS.dataset
    cuda_name = f'cuda:{FLAGS.cuda_id}'
    TRAIN_BATCH_SIZE = FLAGS.batch_size
    TEST_BATCH_SIZE = FLAGS.batch_size
    NUM_EPOCHS = FLAGS.num_epochs
    LR = FLAGS.lr
    Architecture = [BGNN][FLAGS.model]
    model_name = Architecture.__name__
    fold = FLAGS.fold
    if not FLAGS.weighted:
        model_name += "-noweight"
    if fold != -100:
        model_name += f"-{fold}"

    print("Dataset:", dataset)
    print("Cuda name:", cuda_name)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LR)
    print("Model name:", model_name)
    print("Train and test") if fold == -100 else print("Fold of 5-CV:", fold)
    # 创建模型的保存路径
    if os.path.exists(f"models/architecture/{dataset}/S1/cross_validation/") is False:
        os.makedirs(f"models/architecture/{dataset}/S1/cross_validation/")
    if os.path.exists(f"models/predictor/{dataset}/S1/cross_validation/") is False:
        os.makedirs(f"models/predictor/{dataset}/S1/cross_validation/")
    if os.path.exists(f"models/architecture/{dataset}/S1/test/") is False:
        os.makedirs(f"models/architecture/{dataset}/S1/test/")
    if os.path.exists(f"models/predictor/{dataset}/S1/test/") is False:
        os.makedirs(f"models/predictor/{dataset}/S1/test/")

    print("create dataset ...")
    # 从文件中读取亲和力
    affinity = read_data(dataset)
    # print("affinity.shape=", str(affinity.shape))#(68,442)
    # print("affinity=",str(affinity))

    # 返回训练集、测试集以及药物-靶点亲和性图。
    train_data, test_data, affinity_graph = create_dataset_for_train_test(affinity, dataset, fold, FLAGS.weighted,
                                                                          FLAGS.drug_aff_k, FLAGS.target_aff_k)
    print("create train_loader and test_loader ...")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    print("create drug_graphs_dict and target_graphs_dict ...")
    drug_graphs_dict = getDrugMolecularGraph(
        json.load(open(f'data/{dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    target_graphs_dict = getTargetMolecularGraph(
        json.load(open(f'data/{dataset}/targets.txt'), object_pairs_hook=OrderedDict), dataset)
    print("create drug_graphs_DataLoader and target_graphs_DataLoader ...")
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=affinity_graph.num_node1s)  # if memory is not enough, turn down the batch_size, e.g., batch_size=30
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=affinity_graph.num_node2s)  # if memory is not enough, turn down the batch_size, e.g., batch_size=100

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    architecture = Architecture(ag_init_dim=affinity_graph.num_node1s + affinity_graph.num_node2s + 2,
                                affinity_dropout_rate=FLAGS.dropedge_rate, skip=FLAGS.skip)
    architecture.to(device)

    predictor = Predictor(embedding_dim=architecture.output_dim)
    predictor.to(device)

    if fold != -100:
        best_result = [1000]
    print("start training ...")

    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 指定目录和文件名
    directory = "models/evaluate"
    file_name = f"{current_time}.csv"
    file_path = os.path.join(directory, file_name)
    # 创建CSV文件并写入数据
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "MSE", "C-Index", "RM2", "Pearson", "AUPR"])

    patience = 100
    counter = 0

    # 假设您要保存到的文件名为 "output.txt"
    output_file_path = "G_P_output.txt"

    for epoch in range(NUM_EPOCHS):
        train(architecture, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, LR,
              epoch + 1, TRAIN_BATCH_SIZE, affinity_graph)
        G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader,
                          target_graphs_DataLoader, affinity_graph)

        # # 打开文件以追加写入
        # with open(output_file_path, "a") as file:
        #     # 写入当前轮数信息
        #     file.write("Epoch: " + str(epoch) + "\n")
        #     # 将 G 和 P 写入文件
        #     file.write("G:\n")
        #     file.write(str(G) + "\n\n")
        #     file.write("P:\n")
        #     file.write(str(P) + "\n\n")

        # 用于评估模型在给定数据集上的性能，返回一个结果。
        result = model_evaluate(G, P, dataset)
        print("Epoch:" + str(epoch) + " result[mse, cindex, rm2, pearson, aupr=]" + str(result))
        print(type(result))
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            result = (epoch + 1,) + result
            # values = list(result.values())
            writer.writerow(result)

        # 训练和保存最佳模型：在一个循环中，对每个交叉验证折（fold）进行训练和评估。
        # 如果fold 不是特殊值 - 100，并且当前模型的性能优于之前的最佳性能（result[0] < best_result[0]），则更新最佳结果，并保存当前模型的权重。
        # 模型的权重保存到
        # models / architecture / {dataset} / S1 / cross_validation / 目录下。
        # 预测器的权重保存到
        # models / predictor / {dataset} / S1 / cross_validation / 目录下。
        if fold != -100 and result[0] < best_result[0]:
            best_result = result
            best_epoch = epoch + 1
            checkpoint_dir = f"models/architecture/{dataset}/S1/cross_validation/"
            checkpoint_path = checkpoint_dir + model_name + ".pkl"
            torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

            checkpoint_dir = f"models/predictor/{dataset}/S1/cross_validation/"
            checkpoint_path = checkpoint_dir + model_name + ".pkl"
            torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)
        elif fold != -100 and result[0] > best_result[0]:
            # print('No improvement since epoch ', best_epoch, '; best_result',best_result, dataset, fold)

            counter += 1
            if counter > patience:
                break

    # 在测试集上进行预测和评估：如果fold 的值为 - 100，表示执行测试集上的预测和评估。
    # 在这种情况下，模型的权重保存到
    # models / architecture / {dataset} / S1 / test / 和
    # models / predictor / {dataset} / S1 / test / 目录下，并输出测试集上的评估结果。
    if fold == -100:
        checkpoint_dir = f"models/architecture/{dataset}/S1/test/"
        checkpoint_path = checkpoint_dir + model_name + ".pkl"
        torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        checkpoint_dir = f"models/predictor/{dataset}/S1/test/"
        checkpoint_path = checkpoint_dir + model_name + ".pkl"
        torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        # 打印最佳结果或测试结果：根据执行的情况，打印最佳交叉验证结果或测试集结果。
        print('\npredicting for test data')
        G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader,
                          target_graphs_DataLoader, affinity_graph)
        result = model_evaluate(G, P, dataset)
        print("reslut:", result)
    else:
        print(f"\nbest result for fold {fold} of cross validation:")
        print("reslut:", best_result)


if __name__ == '__main__':
    train_test()
