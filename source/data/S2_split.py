import random, json, sys
from collections import OrderedDict
from rdkit import Chem


def split(dataset):
    drugs = json.load(open(f'{dataset}/drugs.txt'), object_pairs_hook=OrderedDict)

    drug_count = 0
    smile_dict = {}
    for d in drugs:
        #使用 RDKit 将分子结构表示（SMILES）转换为规范的 SMILES 表示，
        #并建立一个字典（smile_dict）来跟踪每个 SMILES 对应的药物索引。
        drug = Chem.MolToSmiles(Chem.MolFromSmiles(drugs[d]), isomericSmiles=True)
        if drug not in smile_dict:
            smile_dict[drug] = [drug_count]
        else:
            smile_dict[drug].append(drug_count)
        drug_count += 1

    #创建两个列表：smile_list 存储单一 SMILES 的药物索引，
    # smile_list_bak 存储具有相同 SMILES 的药物索引。
    smile_list = []
    smile_list_bak = []
    for drug in smile_dict:
        if len(smile_dict[drug]) > 1:
            smile_list_bak.extend(smile_dict[drug])
        else:
            smile_list.append(smile_dict[drug][0])

    #对 smile_list 进行随机打乱。
    random.seed(1)
    random.shuffle(smile_list)
    
    k_folds = 6
    CV_size = int(len(smile_list) / k_folds)
    print(drug_count, len(smile_list), len(smile_list_bak), CV_size)
    test_fold = smile_list[:CV_size]#将第一个折叠作为测试集
    print("Num of test fold:", len(test_fold))
    train_folds = []
    for i in range(1, k_folds - 1):#遍历并将其余折叠存储在 train_folds 列表中
        train_folds.append(smile_list[i * CV_size: (i + 1) * CV_size])
        print(f"Num of train fold {i}:", len(train_folds[-1]))
    train_folds.append(smile_list[(i + 1) * CV_size:])
    print(f"Num of train fold {i + 1}:", len(train_folds[-1]))
    json.dump(train_folds, open(f'{dataset}/S2_train_set.txt', "w"))
    json.dump(test_fold, open(f'{dataset}/S2_test_set.txt', "w"))
    json.dump(smile_list_bak, open(f'{dataset}/S2_mask_set.txt', "w"))


if __name__ == '__main__':
    #dataset = sys.argv[1]
    dataset = "davis"
    print(dataset)
    split(dataset)
