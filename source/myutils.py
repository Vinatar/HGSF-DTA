import pickle, argparse
import numpy as np
from itertools import chain
import torch
import torch.nn as nn
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset, Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau


# 通过 argparse 模块解析命令行参数，包括数据集名称、GPU ID、训练轮数、批处理大小、学习率等
# 返回包含命令行参数的命名空间
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset for use', default='kiba')#davis
    parser.add_argument('--cuda_id', type=int, help='Cuda for use', default=0)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train',
                        default=1000)  # num_epochs = 200, when conducting the S2, S3 and S4 experiments
    parser.add_argument('--batch_size', type=int, help='Batch size of dataset', default=256)
    parser.add_argument('--lr', type=float, help='Initial learning rate to train', default=0.0001)  # default=0.0005
    parser.add_argument('--model', type=int, help='Model id', default=0)
    parser.add_argument('--fold', type=int, help='Fold of 5-CV', default=-100)
    parser.add_argument('--weighted', help='Whether affinity graph is weighted', action='store_false')
    parser.add_argument('--dropedge_rate', type=float, help='Rate of edge dropout', default=0.2)
    parser.add_argument('--drug_sim_k', type=int, help='Similarity topk of drug',
                        default=2)  # 基于相似性的表示推断方法通过改变sim Kd和sim Kt来影响HGRLDTA w / o FMG的预测性能。
    parser.add_argument('--target_sim_k', type=int, help='Similarity topk of target', default=7)
    parser.add_argument('--drug_aff_k', type=int, help='Affinity topk of drug', default=40)
    parser.add_argument('--target_aff_k', type=int, help='Affinity topk of target',
                        default=150)  # target_aff_k = 90, when conducting the S2 and S4 experiments on the KIBA dataset
    parser.add_argument('--skip', help='Whether the skip connection operation is used', action='store_true')
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS


# 继承自 PyTorch Geometric 的 InMemoryDataset 类。
# DTADataset 用于处理药物-靶点亲和性数据，每个数据样本包括药物和靶点的 ID 以及相应的亲和性值。
# GraphDataset 用于处理分子图的数据，每个数据样本包括图的大小、特征和边的索引。
# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, drug_ids=None, target_ids=None,drug_smiles=None, target_sequences=None, y=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.process(drug_ids, target_ids, drug_smiles, target_sequences, y)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, drug_ids, target_ids,drug_smiles, target_sequences, y):
        data_list = []
        for i in range(len(drug_ids)):
            DTA = DATA.Data(drug_id=torch.IntTensor([drug_ids[i]]), target_id=torch.IntTensor([target_ids[i]]), drug_smile=drug_smiles[drug_ids[i]], target_sequence=target_sequences[target_ids[i]],
                            y=torch.FloatTensor([y[i]]))
            data_list.append(DTA)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for key in graphs_dict:
            size, features, edge_index = graphs_dict[key]
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            GCNData.__setitem__(f'{self.dttype}_size', torch.LongTensor([size]))
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 使用训练集对模型进行训练。
# 采用均方误差损失函数（nn.MSELoss）。
# 采用 Adam 优化器进行参数更新。
def train(architecture, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, LR, epoch,
          TRAIN_BATCH_SIZE, affinity_graph):
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    architecture.train()
    predictor.train()
    LOG_INTERVAL = 50
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chain(architecture.parameters(), predictor.parameters())), lr=LR,
        weight_decay=0)

    # 定义学习率调度器
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=20000, factor=0.5, verbose=True)

    affinity_graph.to(device)  # affinity graph
    # drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs

    # 在每轮开始前定义一个空列表来存储每个批次的损失和学习率
    epoch_data = []
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        drug_embedding, target_embedding, drug_transform_embedding, target_transform_embedding = architecture(affinity_graph, drug_graph_batchs, target_graph_batchs)
        output, _ = predictor(data.to(device), drug_embedding, target_embedding,  drug_transform_embedding, target_transform_embedding)
        #output, _ = predictor(data.to(device))
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()

        # 在每个 batch 后更新学习率
        # scheduler.step(loss)

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * TRAIN_BATCH_SIZE, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()
            ))

        # 将当前批次的损失、学习率和output添加到列表中
        epoch_data.append({'loss': loss.item(), 'learning_rate': optimizer.param_groups[0]['lr'], 'output': output})

    # 每轮结束后将损失、学习率和output列表保存到文件中
    with open("training_data.txt", "a") as file:
        file.write("Epoch {}\n".format(epoch))
        for data in epoch_data:
            file.write(
                "Loss: {:.6f}\tLearning Rate: {:.6f}\t".format(data['loss'], data['learning_rate']))


# 用于在给定数据集上进行预测。
# 返回模型的预测结果和真实标签。
def predicting(architecture, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader,
               affinity_graph, drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):
    architecture.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    # print('Make prediction for {} samples...'.format(len(loader.dataset)))
    affinity_graph.to(device)  # affinity graph
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        for data in loader:
            drug_embedding, target_embedding,  drug_transform_embedding, target_transform_embedding = architecture(
                affinity_graph, drug_graph_batchs, target_graph_batchs,
                drug_map=drug_map, drug_map_weight=drug_map_weight, target_map=target_map,
                target_map_weight=target_map_weight
            )
            # print("predicting中的drug_embedding:",str(drug_embedding))
            # print("predicting中的target_embedding:", str(target_embedding))
            output, _ = predictor(data.to(device), drug_embedding, target_embedding,  drug_transform_embedding, target_transform_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


# 获取链接嵌入，用于表示药物-靶点亲和性的嵌入向量。
def getLinkEmbeddings(architecture, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader,
                      affinity_graph, drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):
    architecture.eval()
    predictor.eval()
    affinity_graph.to(device)  # affinity graph
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        link_embeddings_batch_list = []
        for data in loader:
            drug_embedding, target_embedding = architecture(
                affinity_graph, drug_graph_batchs, target_graph_batchs,
                drug_map=drug_map, drug_map_weight=drug_map_weight, target_map=target_map,
                target_map_weight=target_map_weight
            )
            _, link_embeddings_batch = predictor(data.to(device), drug_embedding, target_embedding)
            link_embeddings_batch_list.append(link_embeddings_batch.cpu().numpy())
    link_embeddings = np.concatenate(link_embeddings_batch_list, axis=0)
    return link_embeddings


# 获取药物和靶点的嵌入向量。
def getEmbeddings(architecture, device, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph):
    architecture.eval()
    affinity_graph.to(device)  # affinity graph
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        drug_embedding, target_embedding = architecture(affinity_graph, drug_graph_batchs, target_graph_batchs)
    return drug_embedding.cpu().numpy(), target_embedding.cpu().numpy()


# 数据加载和处理：
#
# 使用 PyTorch Geometric 中的 Batch 类进行数据加载和处理。
# 提供了 collate 函数，用于将数据列表转换为 Batch 对象
def collate(data_list):
    batch = Batch.from_data_list(data_list)
    return batch


# 从文件中读取亲和性数据，如药物-靶点亲和性数据。
# 对于数据集 'davis'，将亲和性值转换为负对数形式
def read_data(dataset):
    dataset_path = 'data/' + dataset + '/'
    affinity = pickle.load(open(dataset_path + 'affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    return affinity
