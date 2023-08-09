import time
import pandas as pd
import networkx as nx
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from numba import cuda
from pandas import DataFrame
from sklearn import metrics
from scipy.spatial.distance import squareform
from sknetwork.clustering import get_modularity
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import f1_score

from utils.data_processor import normalize_adj
import clustering



@cuda.jit
def potential_gpu(y, num, dense_matrix, Delta):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < num:
        sum1 = 0
        sum2 = 0
        for i in range(num):
            sum1 += (dense_matrix[idx][i] ** 2) * math.exp((-dense_matrix[idx][i] ** 2) / (2 * (Delta ** 2)))
            sum2 += math.exp((-dense_matrix[idx][i] ** 2) / (2 * (Delta ** 2)))
        y[idx] = (1 / (2 * (Delta ** 2))) * (sum1 / sum2)


@cuda.jit
def gradient_decent_step_one(y, adj_list, num, label):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    index_flag = idx
    if idx < num:
        neighbor = adj_list[idx]
        low_potential_point = y[idx]
        for item in neighbor:
            if item == -1:
                break
            if low_potential_point > y[int(item)]:
                index_flag = item
                low_potential_point = y[int(item)]
    label[idx] = label[int(index_flag)]

def find(p, parent):
    if p != parent[p]:
        parent[p] = find(parent[p], parent)
    return parent[p]

def find_iter(p, parent):
    x = p
    tmp = x
    while p != parent[p]:
        p = parent[p]
    while x != parent[x]:
        tmp = parent[x]
        parent[x] = p
        x = tmp

def gradient_descent_gpu(delta, g, weight="weight", disconnect_weight=1000):
    id = list(nx.to_pandas_adjacency(g))

    label = list(range(len(nx.nodes(g))))
    threads_per_block = 128
    blocks_per_grid = math.ceil(len(nx.nodes(g)) / threads_per_block)
    y = np.zeros((len(nx.nodes(g)), 1))
    dense_matrix = nx.to_numpy_array(g, weight=weight)
    dense_matrix = dense_matrix[:len(nx.nodes(g)), :len(nx.nodes(g))]

    disconnect_pos = dense_matrix == 0
    # dense_matrix = np.divide(1.0, dense_matrix)
    dense_matrix[disconnect_pos] = disconnect_weight

    row, col = np.diag_indices_from(dense_matrix)
    dense_matrix[row, col] = 0
    potential_gpu[blocks_per_grid, threads_per_block](y, len(label), dense_matrix, delta)
    cuda.synchronize()
    adj_array = np.ones((len(label), len(label))) * -1
    mapping_nodes = nx.nodes(g)
    g = nx.convert_node_labels_to_integers(g)
    for i in range(len(label)):
        j = 0
        for item in list(g.adj[i]):
            if item == '-1':
                break
            adj_array[i][j] = item
            j += 1
    label = np.array(label)
    # 转化为向量
    y = y.ravel()
    # print(y)
    gradient_decent_step_one[blocks_per_grid, threads_per_block](y, adj_array, len(label), label)
    start_time = time.perf_counter()
    
    label_dup = list(set(label))
    k = 0
    """
    for item in label_dup:
        while True:
            if label[item] == item:
                break
            else:
                label = [label[item] if i == item else i for i in label]
                label_dup[k] = label[item]
                item = label[item]
        k = k + 1
    """
    # 新梯度下降
    for i, elem in enumerate(label):
            if i != elem: find_iter(i, label)
    
    # print(y)
    # print(label)
    print(len(label))
    print("time on GGD: ", time.perf_counter() - start_time)
    print(len(list(set(label))))
    # print(pd.value_counts(label))
    print("Clusters num: " + str(len(list(set(label)))))
    return id, label, mapping_nodes


def floyd_dist(g):
    shortest_path = nx.floyd_warshall(g)
    dis_mat = np.ones((len(g), len(g)))
    for i in range(len(g)):
        for j in range(len(g)):
            try:
                dis_mat[i][j] = shortest_path[i][j]
            except:
                dis_mat[i][j] = 1000
    # shortest_path = pd.DataFrame(shortest_path).T.fillna(0).values
    # shortest_path[np.isinf(shortest_path)] = 1000
    np.save("dis_mat.npy", dis_mat)
    return dis_mat


def map_label(label_true, label_pred):
    l1 = np.unique(label_true)
    len1 = len(label_true)
    l2 = np.unique(label_pred)
    len2 = len(label_pred)
    ncls = len2
    label = np.zeros((1, ncls))
    for k in range(len2):
        ind = label_pred == k
        tmp = np.zeros((1, ncls))
        for j in range(ncls):
            tmp[j] = np.sum(label_true[ind] == j)
        tmp1 = tmp.tolist()
        l = tmp1.index(max(tmp))
        label[k] = l
    return label


def result_visualization(label, mapping_nodes):
    # 单个数据
    num = len(label)
    label_unique = list(set(label))
    label_num = [0 for i in range(len(label_unique))]
    sub = 0
    for item in label_unique:
        for i in range(len(label)):
            if label[i] == item:
                label_num[sub] = label_num[sub] + 1
        sub = sub + 1
    labels = []
    for item in label_unique:
        labels.append(list(mapping_nodes)[item])
    sizes = [label_num[i] / num * 100 for i in range(len(label_unique))]
    # # 各区域颜色
    # colors = ['red', 'orange', 'yellow', 'green', 'purple', 'blue', 'black']
    # # 数据计算处理
    # sizes = [data[0] / Num * 100, data[1] / Num * 100, data[2] / Num * 100, data[3] / Num * 100, data[4] / Num * 100,
    #          data[5] / Num * 100, data[6] / Num * 100]
    # # 设置突出模块偏移值
    # expodes = (0, 0, 0.1, 0, 0, 0, 0)
    # # 设置绘图属性并绘图
    plt.pie(sizes, shadow=True)
    # ## 用于显示为一个长宽相等的饼图
    plt.axis('equal')
    plt.show()


def label_fill(label):
    dict_data = {"color": list(label)

                 }
    data = DataFrame(dict_data)
    DataFrame(data).to_excel(path_doctor_who_label)


def output_id_and_label(id, label, output_path="output.csv"):
    data = {"Id": id, "Label": label}
    data = pd.DataFrame(data)
    data.to_csv(output_path, sep=',', index=False, header=True)


def read_doctorwho(path):
    weight = "Weight"
    data = pd.read_csv(path)
    data["Weight"] = 1.0 / data["Weight"]
    g = nx.from_pandas_edgelist(data, "Source", "Target", "Weight")
    return g, weight


def load_amac_dataset():
    adj = np.load("./amac/amac_adj.npy")
    feat = np.load("./amac/amac_feat.npy")
    label = np.load("./amac/amac_label.npy")
    g = nx.from_numpy_array(adj)
    return g, label


def load_cora_dataset():
    adj = np.load("./cora/cora_adj.npy")
    feat = np.load("./cora/cora_feat.npy")
    label = np.load("./cora/cora_label.npy")
    g = nx.from_numpy_array(adj)
    return g, label


def load_citeseer_dataset():
    adj = np.load("./citeseer/citeseer_adj.npy")
    feat = np.load("./citeseer/citeseer_feat.npy")
    label = np.load("./citeseer/citeseer_label.npy")
    g = nx.from_numpy_array(adj)
    return g, label


def load_pubmed_dataset():
    adj = np.load("./datasets/pubmed/pubmed_adj.npy")
    feat = np.load("./datasets/pubmed/pubmed_feat.npy")
    label = np.load("./datasets/pubmed/pubmed_label.npy")
    g = nx.from_numpy_array(adj)
    return g, label


def load_wiki_dataset():
    adj = np.load("./wiki/wiki_adj.npy")
    feat = np.load("./wiki/wiki_feat.npy")
    label = np.load("./wiki/wiki_label.npy")
    g = nx.from_numpy_array(adj)
    return g, label


def load_karate_club():
    g = nx.karate_club_graph()
    label = []
    for i in range(len(g)):
        label.append(0 if g.nodes[i]["club"] == 'Mr. Hi' else 1)
    return g, label


def load_amazon():
    from torch_geometric.datasets import Amazon
    from torch_geometric.utils.convert import to_networkx
    dataset = Amazon(root='./tmp/Amazon/Photo', name='Photo')
    g = to_networkx(dataset.data)
    label = dataset.data.y.numpy().tolist()
    return g, label


def load_coauthor():
    from torch_geometric.datasets import Coauthor
    from torch_geometric.utils.convert import to_networkx
    dataset = Coauthor(root='./tmp/Coauthor/CS', name='CS')
    g = to_networkx(dataset.data)
    label = dataset.data.y.numpy().tolist()
    return g, label


def load_AttributedGraphDataset():
    from torch_geometric.datasets import AttributedGraphDataset
    from torch_geometric.utils.convert import to_networkx
    dataset = AttributedGraphDataset(root='./tmp/AttributedGraphDataset/BlogCatalog', name='BlogCatalog')
    g = to_networkx(dataset.data)
    label = dataset.data.y.numpy().tolist()
    return g, label


def load_CitationFull():
    from torch_geometric.datasets import CitationFull
    from torch_geometric.utils.convert import to_networkx
    dataset = CitationFull(root='./tmp/CitationFull/Cora_ML', name='Cora_ML')
    g = to_networkx(dataset.data)
    label = dataset.data.y.numpy().tolist()
    return g, label


def load_Entities():
    from torch_geometric.datasets import Entities
    from torch_geometric.utils.convert import to_networkx
    dataset = Entities(root='./tmp/Entities/AIFB', name='AIFB')
    g = to_networkx(dataset.data)
    label = dataset.data.y.numpy().tolist()
    return g, label


def evaluation(label_true, label_pred, adj):
    true_cluster_num = len(set(label_true))
    pred_cluster_num = len(set(label_pred))
    modularity = get_modularity(adj, np.array(label_pred))
    print("Modularity: " + str(modularity))
    if true_cluster_num == pred_cluster_num:
        acc, nmi, ari, f1 = clustering.evaluation(label_true, label_pred)
        print("Acc: %f\tNMI: %f\tARI: %f\tF1: %f" % (acc, nmi, ari, f1))
    else:
        nmi = normalized_mutual_info_score(label_true, label_pred)
        print("NMI: " + str(nmi))
    rand_score = metrics.adjusted_rand_score(label_true, label_pred)
    print("Rand_score: " + str(rand_score))
    fowlkes_mallows_score = metrics.fowlkes_mallows_score(label_true, label_pred)
    print("Fowlkes_mallows_score: " + str(fowlkes_mallows_score))


def community2label(com, noedenum):
    label = np.zeros(noedenum)
    i = 0
    for set in com:
        for elem in set:
            label[elem] = i
        i = i + 1
    return label


def other_algorithms(g, label_true, weight="weight"):
    from sklearn.cluster import KMeans
    from sknetwork.clustering import Louvain
    from sknetwork.clustering import PropagationClustering
    from sklearn.cluster import SpectralClustering
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.cluster import Birch
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import OPTICS
    adj = nx.to_numpy_array(g, weight=weight)

    print("\n- Kmeans Method:")
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(adj)
    label_pred = kmeans.labels_.tolist()
    evaluation(label_true, label_pred, adj)

    print("\n- Louvain's Method:")
    louvain = Louvain()
    label_pred = louvain.fit_transform(adj)
    evaluation(label_true, label_pred, adj)

    print("\n- Local and Global Consistency(LGC):")
    from networkx.algorithms import node_classification
    predicted = node_classification.local_and_global_consistency(g, label_name='club')
    propagation = PropagationClustering()
    label_pred = propagation.fit_transform(adj)
    print(predicted)
    print(label_pred)
    evaluation(label_true, predicted, adj)

    print("\n- Spectral Clustering:")
    clustering = SpectralClustering(
                                    affinity='precomputed',
                                    assign_labels='kmeans',
                                    random_state=0).fit(adj)
    label_pred = clustering.labels_.tolist()
    evaluation(label_true, label_pred, adj)

    print("\n- AGNES:")
    clustering = AgglomerativeClustering().fit(adj)
    label_pred = clustering.labels_.tolist()
    evaluation(label_true, label_pred, adj)

    print("\n- BIRCH:")
    brc = Birch(n_clusters=None).fit(adj)
    label_pred = brc.labels_.tolist()
    evaluation(label_true, label_pred, adj)

    print("\n- DBSCAN:")
    db = DBSCAN(metric='precomputed', eps=10).fit(adj)
    label_pred = db.labels_.tolist()
    # print(label_pred)
    evaluation(label_true, label_pred, adj)

    print("\n- OPTICS:")
    optic = OPTICS(metric='precomputed').fit(adj)
    label_pred = optic.labels_.tolist()
    # print(label_pred)
    evaluation(label_true, label_pred, adj)


if __name__ == '__main__':
    # label, mapping_nodes = gradient_descent_gpu(delta=200, path=path_darwin)
    # result_visualization(label, mapping_nodes)

    # g, weight = read_doctorwho(path_doctor_who)
    # id, label, mapping_nodes = gradient_descent_gpu(delta=156, g=g, weight=weight)
    # dis = floyd_dist(g)

    # g, label_true = load_cora_dataset()
    # id, label, mapping_nodes = gradient_descent_gpu(delta=1e8, g=g)
    

    # g, label_true = load_citeseer_dataset()
    # id, label, mapping_nodes = gradient_descent_gpu(delta=300, g=g)

    g, label_true = load_pubmed_dataset()
    id, label, mapping_nodes = gradient_descent_gpu(delta=1e12, g=g, disconnect_weight=1e5)

    # g, label_true = load_wiki_dataset()
    # id, label, mapping_nodes = gradient_descent_gpu(delta=500, g=g)


    # g, label_true = load_karate_club()
    # id, label, mapping_nodes = gradient_descent_gpu(delta=500, g=g)
    

    # g, label_true = load_coauthor()
    # id, label, mapping_nodes = gradient_descent_gpu(delta=500, g=g)


    # g, label_true = load_CitationFull()
    # id, label, mapping_nodes = gradient_descent_gpu(delta=200, g=g)
    # nx.write_gml(g, "coraml.gml")

    # dis = floyd_dist(g)
    """
    test example
    a = [[0, 1, 0, 0, 1, 0],
         [1, 0, 0, 0, 1, 1],
         [0, 0, 0, 1, 0, 1],
         [0, 0, 1, 0, 0, 1],
         [1, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 0, 0]]
    g = nx.from_numpy_array(np.array(a))
    dis = floyd_dist(g)
    """

    # dis = np.load("dis_mat.npy")
    # dis = np.load("cora_dis.npy")
    # dis[dis == np.inf] = 1000
    # dis = normalize(dis, norm="l2")
    # dis = normalize_adj(dis, symmetry=False)

    # label_fill(label)
    # output_id_and_label(id, label_true, output_path='cora_nodes.csv')
    # result_visualization(label, mapping_nodes)

    # np.fill_diagonal(dis, 0)
    # silhouette_score = metrics.silhouette_score(dis, label, metric="precomputed")
    # print(silhouette_score)

    # print("~QC algorithm:")
    # evaluation(label_true, label, nx.to_numpy_array(g))

    # other_algorithms(g, label_true)
