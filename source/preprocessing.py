import numpy as np
import pandas as pd
import torch
import re


VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
				"U": 19, "T": 20, "W": 21,
				"V": 22, "Y": 23, "X": 24,
				"Z": 25 }

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

def seqs2int(target):

    return [VOCAB_PROTEIN[s] for s in target.upper()]

def process_target_data(data_path):
    # 使用pd.read_csv(data_path)函数读取CSV文件，将文件中的数据加载到DataFrame对象df中。
    #df = pd.read_csv(data_path)
    # 打开文件并逐行读取内容
    with open(data_path, 'r') as file:
        text = file.read()
    # 创建两个空列表：data_list用于存储处理后的数据
    target_list = []

    # 使用正则表达式提取键值对
    pattern = r'\{([^}]*)\}'
    matches = re.findall(pattern, text)

    # 遍历匹配结果并处理键值对0
    for match in matches:
        # 使用逗号分隔键值对
        pairs = match.split(',')
        for pair in pairs:
            # 解析键值对
            #print(pair)
            key, sequence = pair.strip().split(":")

            # 去掉值中的双引号
            sequence = sequence.strip().strip('"')

            # 将目标序列sequence转换为整数编码的向量，并进行长度填充或截断，使其长度为1200。然后，将其存储为PyTorch张量。
            target = seqs2int(sequence)
            #target_len = 1200#davis
            target_len = 1200  # kiba
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]


            target=torch.LongTensor([target])

            target_list.append(target)

    return target_list


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()
def process_drug_data(data_path):
    # 使用pd.read_csv(data_path)函数读取CSV文件，将文件中的数据加载到DataFrame对象df中。
    # df = pd.read_csv(data_path)
    # 打开文件并逐行读取内容
    with open(data_path, 'r') as file:
        text = file.read()
    # 创建两个空列表：data_list用于存储处理后的数据
    drug_list = []

    # 使用正则表达式提取键值对
    pattern = r'\{([^}]*)\}'
    matches = re.findall(pattern, text)

    # 遍历匹配结果并处理键值对
    for match in matches:
        # 使用逗号分隔键值对
        pairs = match.split(',')
        for pair in pairs:
            # 解析键值对
            # print(pair)
            key, smiles = pair.strip().split(":")

            # 去掉值中的双引号
            smiles = smiles.strip().strip('"')

            #drug = label_smiles(smiles, 85, CHARISOSMISET)#davis
            drug = label_smiles(smiles, 100, CHARISOSMISET)  # kiba

            drug = torch.LongTensor([drug])
            drug_list.append(drug)

    return drug_list

# SMILES 的固定最大长度为 85，Davis 的蛋白质序列固定最大长度为 1200。
# KIBA 的组成部分，我们为 SMILES 选择了最大 100 个字符长度，为蛋白质序列选择了 1200 个字符长度。
if __name__ == "__main__":
    dataset = "davis"
    target_data_path = 'data/' + dataset + '/targets.txt'
    #target_list = process_target_data(target_data_path)
    # 访问第一个目标序列
    # first_target = target_list[0]
    # print("First target sequence:", first_target)
    dataset = "davis"
    drug_data_path = 'data/' + dataset + '/drugs.txt'
    drug_list = process_drug_data(drug_data_path)