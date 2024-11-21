import torch
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv, DenseGCNConv, GATConv, SAGPooling, global_mean_pool as gep, \
    global_max_pool as gmp
from torch_geometric.utils import dropout_adj, dropout_edge
import torch.nn.functional as F
import math
from torch_geometric.data import Data
import numpy as np
from layers import (
    IntraGraphAttention,
)
from einops.layers.torch import Reduce
from collections import OrderedDict
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 这个字典中的预测函数（combination）可以执行不同的操作，例如拼接、加法、减法、乘法等。这些操作可以用于联合药物和目标的特征，生成最终的输入特征用于模型的预测
vector_operations = {
    "cat": (lambda x, y: torch.cat((x, y), -1), lambda dim: 2 * dim),
    "add": (torch.add, lambda dim: dim),
    "sub": (torch.sub, lambda dim: dim),
    "mul": (torch.mul, lambda dim: dim),
    "combination1": (lambda x, y: torch.cat((x, y, torch.add(x, y)), -1), lambda dim: 3 * dim),
    "combination2": (lambda x, y: torch.cat((x, y, torch.sub(x, y)), -1), lambda dim: 3 * dim),
    "combination3": (lambda x, y: torch.cat((x, y, torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination4": (lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 2 * dim),
    "combination5": (lambda x, y: torch.cat((torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination6": (lambda x, y: torch.cat((torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination7": (
    lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination8": (lambda x, y: torch.cat((x, y, torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination9": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination10": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 4 * dim),
    "combination11": (
    lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 5 * dim)
}


class LinearBlock(torch.nn.Module):
    # linear_layers_dim：一个包含线性层维度的列表，用于指定每一层的输入和输出维度。[affinity_graph_dims[-1], 1024, drug_graph_dims[1]]
    # dropout_rate：丢弃率，表示在丢弃层中应该丢弃的输入比例。0.1,
    # relu_layers_index：一个包含应用ReLU激活函数的层的索引列表。relu_layers_index=[0],
    # dropout_layers_index：一个包含应用丢弃层的索引列表。  dropout_layers_index=[0, 1]
    def __init__(self, linear_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    # 接受输入张量 x，通过一系列线性层、ReLU激活函数和可选的丢弃层，计算并返回模块的输出。
    # 还将每一步的输出作为嵌入（embeddings）保存在列表中。
    def forward(self, x):
        output = x
        embeddings = [x]
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)
        return embeddings


class DenseGCNBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[],
                 supplement_mode=None):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                conv_layer_input = supplement_dim_func(gcn_layers_dim[i])
            else:
                conv_layer_input = gcn_layers_dim[i]
            conv_layer = DenseGCNConv(conv_layer_input, gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj, supplement_x=None):
        output = x
        embeddings = [x]
        for conv_layer_index in range(len(self.conv_layers)):
            if supplement_x is not None and conv_layer_index == 1:
                supplement_x = torch.unsqueeze(supplement_x, 0)
                output = self.supplement_func(output, supplement_x)
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))
        return embeddings


# ，GCNBlock 是图卷积网络中的一个层或模块，它允许在图卷积层之间应用ReLU激活函数和丢弃层。
# 此外，如果提供了额外的辅助信息，会在第二个图卷积层（conv_layer_index == 1）进行处理。
class GCNBlock(torch.nn.Module):
    # gcn_layers_dim：一个包含图卷积层维度的列表，用于指定每一层的输入和输出维度。drug=[78,78,156,312],target=[54,54,108,216]
    # dropout_rate：丢弃率，表示在丢弃层中应该丢弃的输入比例。                 drug=[78,64,128,256],target=[54,64,128,256]
    # relu_layers_index：一个包含应用ReLU激活函数的层的索引列表。{0,1,2}
    # dropout_layers_index：一个包含应用丢弃层的索引列表。
    # supplement_mode：用于指定是否使用额外的辅助信息的模式。
    def __init__(self, gcn_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[],
                 supplement_mode=None,
                 gamma: float = 0.0001,
                 eta: float = 1):
        super(GCNBlock, self).__init__()

        self.conv_layers = torch.nn.ModuleList()

        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                conv_layer_input = supplement_dim_func(gcn_layers_dim[i])
            else:
                conv_layer_input = gcn_layers_dim[i]
            conv_layer = GCNConv(conv_layer_input, gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

        self.nnl = Linear(gcn_layers_dim[0] , 128)
        #self.feature_conv = GATConv(gcn_layers_dim[0], gcn_layers_dim[0] / 2, 2)
        self.intraAtt = IntraGraphAttention(gcn_layers_dim[2])

        self.pool1 = SAGPooling(7 * gcn_layers_dim[1] + gcn_layers_dim[0], ratio=1.0, GNN=GCNConv)
    # 接受输入x（节点特征）、edge_index（边的索引）、edge_weight（边的权重）、batch（批处理信息）以及可选的额外辅助信息 supplement_x。
    # 在前向传播中，通过调用GCNConv进行图卷积操作。
    def forward(self, x, edge_index, edge_weight, batch, supplement_x=None):

        result = []
        output = x
        embeddings = [x]
        for conv_layer_index in range(len(self.conv_layers)):
            if conv_layer_index == 0:
                # 78->128 54->128
                output = self.relu(self.nnl(output))
                # 128->64
                output = self.intraAtt(Data(output, edge_index))
            if supplement_x is not None and conv_layer_index == 1:
                # 64+64->128
                output = self.supplement_func(output, supplement_x)
            if conv_layer_index != 0:
                # 128->128 128->256
                output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            # 将当前的输出添加到结果列表中
            result.append(output)#64->128->256
            # gep:将所有节点特征取平均值作为图的表示
            embeddings.append(gep(output, batch))
        # 在维度1上连接结果列表中的所有张量
        result = torch.cat(result, dim=1)  # 64 + 128 + 256 = 64 * 7 = 448
        #x, edge_index, _, batch, _, _ = self.pool1(result, edge_index, None, batch)
        #x = torch.cat([gmp(x, batch), gep(x, batch)], dim=1)
        x = gep(result, batch)
        return x#Tensor(68,448)


# 这个模块的作用是在图数据上执行稠密图卷积操作，并生成嵌入表示。
class DenseGCNModel(torch.nn.Module):
    # layers_dim：一个包含图卷积层维度的列表，指定输入、输出和中间层的维度。[ag_init_dim, 512, 256]
    # edge_dropout_rate：边的dropout率，用于随机删除图中的边。0.2
    # supplement_mode：用于指定图卷积操作中的辅助模式（可能是一种集成策略）的参数
    def __init__(self, layers_dim, edge_dropout_rate=0, supplement_mode=None):
        super(DenseGCNModel, self).__init__()
        print('DenseGCNModel Loaded')

        # edge_dropout_rate 和 num_layers 被存储为模型的属性。
        # DenseGCNBlock 被实例化为 graph_conv 属性，用于执行具体的稠密图卷积操作
        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=range(self.num_layers),
                                        dropout_layers_index=range(self.num_layers), supplement_mode=supplement_mode)

    def forward(self, graph, substitution_x=None, supplement_x=None):
        # 接收一个图对象 graph 作为输入，该图包括节点特征 graph.x、邻接矩阵 graph.adj，以及节点的数量信息 num_node1s 和 num_node2s。
        # 如果提供了替代的节点特征 substitution_x 或辅助特征 supplement_x，则使用它们替代默认的 graph.x。
        xs, adj, num_node1s, num_node2s = (
            substitution_x if substitution_x is not None else graph.x), graph.adj, graph.num_node1s, graph.num_node2s
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        # 利用 dropout_adj 函数对邻接矩阵进行边的dropout操作，以防止过拟合。
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_node1s + num_node2s,
                                                                training=self.training)
        # edge_indexs_dropout, edge_weights_dropout = dropout_edge(edge_index=edge_indexs,
        #                                                         p=self.edge_dropout_rate, force_undirected=True,
        #                                                         training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        # 使用 DenseGCNBlock 中的图卷积操作进行前向传播，生成图的嵌入 embeddings。
        embeddings = self.graph_conv(xs, adj_dropout, supplement_x=supplement_x)

        return embeddings


class GCNModel(torch.nn.Module):
    # layers_dim：一个包含图卷积层维度的列表，用于指定每一层的输入和输出维度。
    # supplement_mode：用于指定是否使用额外的辅助信息的模式。
    def __init__(self, layers_dim, supplement_mode=None):
        super(GCNModel, self).__init__()
        print('GCNModel Loaded')

        self.num_layers = len(layers_dim) - 1
        # GCNBlock 是图卷积网络中的一个层或模块，它允许在图卷积层之间应用ReLU激活函数和丢弃层。
        self.graph_conv = GCNBlock(layers_dim, relu_layers_index=range(self.num_layers),
                                   supplement_mode=supplement_mode)

    def forward(self, graph_batchs, supplement_x=None):

        if supplement_x is not None:
            supplement_i = 0
            for graph_batch in graph_batchs:
                graph_batch.__setitem__('supplement_x',
                                        supplement_x[supplement_i: supplement_i + graph_batch.num_graphs])
                supplement_i += graph_batch.num_graphs

            embedding_batchs = list(map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch,
                                                                      supplement_x=graph.supplement_x[
                                                                          graph.batch.int().cpu().numpy()]),
                                        graph_batchs))
        else:
            embedding_batchs = list(
                map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))

        # embeddings = []
        # for i in range(self.num_layers + 1):
        #     embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embedding_batchs


class BGNN(torch.nn.Module):
    # 初始化方法
    # ag_init_dim、mg_init_dim、pg_init_dim：亲和图、药物图和靶标图的初始维度。
    # affinity_dropout_rate：亲和力图卷积中的dropout率。
    # skip：一个布尔值，控制是否使用跳跃连接。
    # embedding_dim：最终嵌入的维度。
    # integration_mode：集成模式，默认为"combination4"。
    def __init__(self, ag_init_dim=2339, mg_init_dim=78, pg_init_dim=54, affinity_dropout_rate=0.2, skip=False,
                 embedding_dim=128, integration_mode="combination4", heads_out_feat_params=[64, 64, 64, 64],
                 blocks_params=[2, 2, 2, 2], hidd_dim=128):
        super(BGNN, self).__init__()
        print('ConvNet Loaded')

        # 亲和力图卷积层的维度
        affinity_graph_dims = [ag_init_dim, 512, 256]

        # 药物图和靶标图卷积层的维度
        # drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4]
        # target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4]
        drug_graph_dims = [mg_init_dim, 64, 128, 256]
        target_graph_dims = [pg_init_dim, 64, 128, 256]

        # 药物和靶标嵌入转换线性块的维度
        # drug_transform_dims = [affinity_graph_dims[-1], 1024, drug_graph_dims[1]]
        # target_transform_dims = [affinity_graph_dims[-1], 1024, target_graph_dims[1]]
        drug_transform_dims = [affinity_graph_dims[-1], 1024, 64]
        target_transform_dims = [affinity_graph_dims[-1], 1024, 64]

        # 根据skip的值，定义药物和靶标输出的维度，考虑是否使用跳跃连接 skip=False,
        self.skip = skip
        if not skip:
            # drug_output_dims = [drug_graph_dims[-1], 1024, embedding_dim]
            # target_output_dims = [target_graph_dims[-1], 1024, embedding_dim]
            drug_output_dims = [448, 1024, 128]
            target_output_dims = [448, 1024, 128]
        else:
            drug_output_dims = [512, 1024, 128]
            target_output_dims = [512, 1024, 128]

        self.output_dim = embedding_dim

        # 初始化亲和力图卷积、药物嵌入转换、靶标嵌入转换、药物图卷积、靶标图卷积以及输出线性块
        # DenseGCNModel的作用是在图数据上执行稠密图卷积操作，并生成嵌入表示。
        # inearBlock是构建包含多个线性层的神经网络块，并允许在其中的特定层应用ReLU激活函数和丢弃层。
        self.affinity_graph_conv = DenseGCNModel(affinity_graph_dims, affinity_dropout_rate)
        self.drug_transform_linear = LinearBlock(drug_transform_dims, 0.1, relu_layers_index=[0],
                                                 dropout_layers_index=[0, 1])
        self.target_transform_linear = LinearBlock(target_transform_dims, 0.1, relu_layers_index=[0],
                                                   dropout_layers_index=[0, 1])

        # GCNModel是一个基于图卷积网络的模型，用于处理图数据。通过定义不同层的维度，可以构建具有不同层数和维度的图卷积网络。
        self.drug_graph_conv = GCNModel(drug_graph_dims, supplement_mode=integration_mode)
        self.target_graph_conv = GCNModel(target_graph_dims, supplement_mode=integration_mode)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],
                                                dropout_layers_index=[0, 1])

    # drug_graph_batchs, target_graph_batchs每个批次含有所有的分子/靶标图（68/442）
    def forward(self, affinity_graph, drug_graph_batchs, target_graph_batchs, drug_map=None, drug_map_weight=None,
                target_map=None, target_map_weight=None):

        num_node1s, num_node2s = affinity_graph.num_node1s, affinity_graph.num_node2s

        # 通过亲和力图卷积得到亲和力图的嵌入。
        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]

        # 如果提供了药物和靶标的映射，将亲和力图的嵌入转换成相应的药物和靶标嵌入
        if drug_map is not None:
            if drug_map_weight is not None:
                drug_transform_embedding = torch.sum(
                    self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1][drug_map,
                    :] * drug_map_weight, dim=-2)
            else:
                drug_transform_embedding = torch.mean(
                    self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1][drug_map, :], dim=-2)
        else:
            drug_transform_embedding = self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1]

        if target_map is not None:
            if target_map_weight is not None:
                target_transform_embedding = torch.sum(
                    self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1][target_map,
                    :] * target_map_weight, dim=-2)
            else:
                target_transform_embedding = torch.mean(
                    self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1][target_map, :], dim=-2)
        else:
            target_transform_embedding = self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1]

        # 对药物和靶标进行图卷积
        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs, supplement_x=drug_transform_embedding)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs, supplement_x=target_transform_embedding)[
            -1]

        # 考虑是否使用跳跃连接
        if not self.skip:
            drug_output_embedding = self.drug_output_linear(drug_graph_embedding)[-1]
            target_output_embedding = self.target_output_linear(target_graph_embedding)[-1]
        else:
            drug_output_embedding = \
                self.drug_output_linear(torch.cat((drug_graph_embedding, drug_transform_embedding), 1))[-1]
            target_output_embedding = \
                self.target_output_linear(torch.cat((target_graph_embedding, target_transform_embedding), 1))[-1]

        # 最终通过输出线性块得到药物和靶标的输出嵌入。
        return drug_output_embedding, target_output_embedding, drug_transform_embedding, target_transform_embedding



# 特征融合
class DTF(torch.nn.Module):
    def __init__(self, channels=256, r=4):
        super(DTF, self).__init__()
        inter_channels = int(channels // r)

        self.att1 = torch.nn.Sequential(
            torch.nn.Linear(channels, inter_channels),
            torch.nn.BatchNorm1d(inter_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(inter_channels, channels),
            torch.nn.BatchNorm1d(channels)
        )

        self.att2 = torch.nn.Sequential(
            torch.nn.Linear(channels, inter_channels),
            torch.nn.BatchNorm1d(inter_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(inter_channels, channels),
            torch.nn.BatchNorm1d(channels)
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, fd, fp):
        w1 = self.sigmoid(self.att1(fd + fp))
        fout1 = fd * w1 + fp * (1 - w1)

        w2 = self.sigmoid(self.att2(fout1))
        fout2 = fd * w2 + fp * (1 - w2)
        return fout2






class mutil_head_attention(torch.nn.Module):
    def __init__(self, head=8, conv=32):
        super(mutil_head_attention, self).__init__()
        self.conv = conv
        self.head = head
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.d_a = torch.nn.Linear(self.conv * 3, self.conv * 3 * head)
        self.p_a = torch.nn.Linear(self.conv * 3, self.conv * 3 * head)
        self.scale = torch.sqrt(torch.FloatTensor([self.conv * 3])).cuda()

    def forward(self, drug, protein):
        bsz, d_ef, d_il = drug.shape
        bsz, p_ef, p_il = protein.shape
        drug_att = self.relu(self.d_a(drug.permute(0, 2, 1))).view(bsz, self.head, d_il, d_ef)
        protein_att = self.relu(self.p_a(protein.permute(0, 2, 1))).view(bsz, self.head, p_il, p_ef)
        interaction_map = torch.mean(self.tanh(torch.matmul(drug_att, protein_att.permute(0, 1, 3, 2)) / self.scale), 1)
        Compound_atte = self.tanh(torch.sum(interaction_map, 2)).unsqueeze(1)
        Protein_atte = self.tanh(torch.sum(interaction_map, 1)).unsqueeze(1)
        drug = drug * Compound_atte
        protein = protein * Protein_atte
        return drug, protein


#Transformer Parameters
d_model = 128 #Embedding Size128
d_ff = 512 #FeedForward dimension512
d_k = d_v = 16 #dimension of K(=Q), V16
# n_layers = 1 #number of Encoder
n_heads = 8 #number of heads in Multi-Head Attention
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        #Q: [batch_size, n_heads, len_q, d_k]
        #K: [batch_size, n_heads, len_k, d_k]
        #V: [batch_size, n_heads, len_v(=len_k), d_v]
        #attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) #scores:[batch_size, n_heads, len_q, len_k]

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9) #Fills elements of self tensor with value where mask is True.

        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) #[batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        #d_k = d_v = 16 #dimension of K(=Q), V
        #n_heads = 8 #number of heads in Multi-Head Attention
        #d_model = 128 #Embedding Size
        super(MultiHeadAttention, self).__init__()
        self.fc0 = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_Q = torch.nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = torch.nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = torch.nn.Linear(d_model, d_v*n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.fc = torch.nn.Linear(n_heads*d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        #input_Q: [batch_size, len_q, d_model]
        #input_K: [batch_size, len_k, d_model]
        #input_V: [batch_size, len_v(=len_k), d_model]
        #attn_mask: [batch_size, seq_len, seq_len]

        #batch_size, seq_len, model_len = input_Q.size()
        if attn_mask is not None:
            batch_size, seq_len, model_len = input_Q.size()
            residual_2D = input_Q.view(batch_size*seq_len, model_len)
            residual = self.fc0(residual_2D).view(batch_size, seq_len, model_len)
        else:
            residual, batch_size = input_Q, input_Q.size(0)

        '''residual, batch_size = input_Q, input_Q.size(0)'''
        #(B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) #Q:[bs, heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2) #K:[bs, heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2) #V:[bs, heads, len_v(=len_k), d_v]

        if attn_mask is not None:
            #attn_mask:[batch_size, n_heads, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        #context:[batch_size, n_heads, len_q, d_v]
        #attn:[batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads*d_v) #context:[bs, len_q, heads*d_v]
        output = self.fc(context) #[batch_size, len_q, d_model]
        return torch.nn.LayerNorm(d_model).to('cuda:0')(output + residual), attn
class Inner_MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(Inner_MultiHeadAttention, self).__init__()
        self.W_Q = torch.nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_K = torch.nn.Linear(d_model, d_k*n_heads, bias=False)
        self.W_V = torch.nn.Linear(d_model, d_v*n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        self.fc = torch.nn.Linear(n_heads*d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        #input_Q: [batch_size, d_model]
        #input_K: [batch_size, d_model]
        #input_V: [batch_size, len_v(=len_k), d_model]
        #attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size, len_q = input_Q, input_Q.size(0), input_Q.size(1)

        #input_K1: [batch_size, len_k, d_model]
        #input_V1: [batch_size, len_k, d_model]
        input_K1 = input_K.unsqueeze(1).repeat(1, len_q, 1)
        input_V1 = input_V.unsqueeze(1).repeat(1, len_q, 1)

        #(B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) #Q:[bs, heads, len_q, d_k]
        K = self.W_K(input_K1).view(batch_size, -1, n_heads, d_k).transpose(1, 2) #K:[bs, heads, len_k, d_k]
        V = self.W_V(input_V1).view(batch_size, -1, n_heads, d_v).transpose(1, 2) #V:[bs, heads, len_v(=len_k), d_v]

        #attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) #attn_mask:[bs, heads, seq_len, seq_len]

        #context:[batch_size, n_heads, len_q, d_v]
        #attn:[batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads*d_v) #context:[bs, len_q, heads*d_v]
        output = self.fc(context) #[batch_size, len_q, d_model]
        return torch.nn.LayerNorm(d_model).to('cuda:0')(output + residual), attn


class PoswiseFeedForwardNet(torch.nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        #inputs:[batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return torch.nn.LayerNorm(d_model).to('cuda:0')(output + residual) #[batch_size, seq_len, d_model]
class Inner_EncoderLayer(torch.nn.Module):
    def __init__(self):
        super(Inner_EncoderLayer, self).__init__()
        self.enc_self_attn = Inner_MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, inputs_Q, inputs_K, inputs_V, enc_self_attn_mask):
        #enc_inputs:[batch_size, src_len, d_model]
        #enc_self_attn_mask:[batch_size, src_len, src_len]

        #enc_outputs:[batch_size, src_len, d_model]
        #attn:[batch_size, n_heads, src_len, src_len]
        #enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(inputs_Q, inputs_K, inputs_V, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) #enc_outputs:[batch_size, src_len, d_model]
        return enc_outputs, attn

class Inter_EncoderLayer(torch.nn.Module):
    def __init__(self):
        super(Inter_EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, inputs_Q, inputs_K, inputs_V, enc_self_attn_mask):
        #enc_inputs:[batch_size, src_len, d_model]
        #enc_self_attn_mask:[batch_size, src_len, src_len]

        #enc_outputs:[batch_size, src_len, d_model]
        #attn:[batch_size, n_heads, src_len, src_len]
        #enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(inputs_Q, inputs_K, inputs_V, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) #enc_outputs:[batch_size, src_len, d_model]
        return enc_outputs, attn


class ResDilaCNNBlock(torch.nn.Module):
    def __init__(self, dilaSize, filterSize=256, dropout=0.15, name='ResDilaCNNBlock'):
        super(ResDilaCNNBlock, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
            torch.nn.ReLU(),
            torch.nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
        )
        self.name = name

    def forward(self, x):
        # x: batchSize × filterSize × seqLen
        return x + self.layers(x)
class ResDilaCNNBlocks(torch.nn.Module):
    # def __init__(self, feaSize, filterSize, blockNum=5, dropout=0.35, name='ResDilaCNNBlocks'):
    def __init__(self, feaSize, filterSize, blockNum=5, dilaSizeList=[1, 2, 4, 8, 16], dropout=0.5,
                 name='ResDilaCNNBlocks'):
        super(ResDilaCNNBlocks, self).__init__()  #
        self.blockLayers = torch.nn.Sequential()
        self.linear = torch.nn.Linear(feaSize, filterSize)
        for i in range(blockNum):
            self.blockLayers.add_module(f"ResDilaCNNBlock{i}",
                                        ResDilaCNNBlock(dilaSizeList[i % len(dilaSizeList)], filterSize,
                                                        dropout=dropout))
            # self.blockLayers.add_module(f"ResDilaCNNBlock{i}", ResDilaCNNBlock(filterSize,dropout=dropout))
        self.name = name
        self.act = torch.nn.ReLU()

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.linear(x)  # => batchSize × seqLen × filterSize
        x = self.blockLayers(x.transpose(1, 2))  # => batchSize × seqLen × filterSize
        x = self.act(x)  # => batchSize × seqLen × filterSize

        # x = self.pool(x.transpose(1, 2))
        x = Reduce('b c t -> b c', 'max')(x)
        return x

class Predictor(torch.nn.Module):
    # embedding_dim：嵌入维度，表示输入嵌入的维度。128
    # output_dim：输出维度，表示模型的最终输出维度。1
    # prediction_mode：用于指定预测函数的模式，从vector_operations字典中获取相应的函数。
    # davis: drug85 target1200
    # kiba: drug100 target1000
    def __init__(self, embedding_dim=128, output_dim=1, prediction_mode="cat", protein_MAX_LENGH=1200,
                 protein_kernel=[4, 8, 12],
                 drug_MAX_LENGH=100, drug_kernel=[4, 6, 8],
                 conv=32, char_dim=128, head_num=8, dropout_rate=0.1):
        super(Predictor, self).__init__()
        print('Predictor Loaded')

        self.dim = char_dim
        self.conv = conv
        self.dropout_rate = dropout_rate
        self.head_num = head_num
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = protein_kernel

        self.prediction_func, prediction_dim_func = vector_operations[prediction_mode]
        # mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]
        mlp_layers_dim = [640, 1024, 1024, 512, output_dim]

        self.dtf = DTF()

        # share weights
        self.inner_cross_atten = Inner_EncoderLayer()
        self.inter_cross_atten = Inter_EncoderLayer()

        self.hidden_dim = 128
        self.reg_dim = 1
        self.dropout_ratio = 0.1
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(320, self.hidden_dim * 8, bias=False),
            torch.nn.LayerNorm(self.hidden_dim * 8),
            torch.nn.Dropout(self.dropout_ratio),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(self.hidden_dim * 8, self.hidden_dim * 2, bias=False),
        )

        self.reg_fun = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 4, self.hidden_dim * 16, bias=False),
            torch.nn.LayerNorm(self.hidden_dim * 16),
            torch.nn.Dropout(self.dropout_ratio),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(self.hidden_dim * 16, self.hidden_dim * 4, bias=False),
            torch.nn.LayerNorm(self.hidden_dim * 4),
            torch.nn.Dropout(self.dropout_ratio),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(self.hidden_dim * 4, self.reg_dim, bias=False)
        )


        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

        self.protein_embed = torch.nn.Embedding(26, self.dim, padding_idx=0)
        self.drug_embed = torch.nn.Embedding(65, self.dim, padding_idx=0)
        self.onehot_smi_net = ResDilaCNNBlocks(self.dim, self.dim, name='res_compound')
        self.onehot_prot_net = ResDilaCNNBlocks(self.dim, self.dim, name='res_prot')


    def forward(self, data, drug_embedding, target_embedding, drug_transform_embedding, target_transform_embedding):
        # def forward(self, data):
        # 通过索引 data.drug_id 和 data.target_id 获取对应的药物特征和目标特征。
        drug_id, target_id, drug_smile, target_sequence, y = data.drug_id, data.target_id, data.drug_smile, data.target_sequence, data.y

        drugembed = self.drug_embed(drug_smile)
        proteinembed = self.protein_embed(target_sequence)

        drugembed = self.onehot_prot_net(drugembed)
        proteinembed = self.onehot_smi_net(proteinembed)

        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]

        InnerAtten_outD, _ = self.inner_cross_atten(drugembed, drug_feature, drug_feature, None)
        InnerAtten_outT, _ = self.inner_cross_atten(proteinembed, target_feature, target_feature, None)

        # inter_cross_atten for drug and target
        # T2D_out 和 D2T_out 是在药物和目标之间进行的交叉注意力的结果。
        T2D_out, _ = self.inter_cross_atten(InnerAtten_outD, InnerAtten_outT, InnerAtten_outT, None)
        D2T_out, _ = self.inter_cross_atten(InnerAtten_outT, InnerAtten_outD, InnerAtten_outD, None)

        # 将药物特征和目标特征传递给 self.prediction_func，
        # 该函数从预定义的 vector_operations 字典中获取，以获得联合特征 concat_feature
        #concat_feature = self.prediction_func(pair, pair_conv)
        # seq features plus graph features
        # din 和 tin 是序列表示和图表示拼接的结果
        # dout 和 tout 是分别对 din 和 tin 进行线性投影的结果。
        # DT_out 是 dout 和 tout 拼接的结果。
        drug1= drug_transform_embedding[drug_id.int().cpu().numpy()]
        target1 = target_transform_embedding[target_id.int().cpu().numpy()]
        din = torch.cat((torch.sum(T2D_out, 1), drug_feature, drug1), 1)
        tin = torch.cat((torch.sum(D2T_out, 1), target_feature, target1), 1)

        dout = self.projection(din)
        tout = self.projection(tin)
        DT_out = torch.cat((dout, tout), 1)
        #DT_out = torch.cat((din, tin), 1)
        # affi 是经过一个全连接层（reg_fun）得到的最终模型预测结果。
        affi = self.reg_fun(DT_out)

        return affi, _

