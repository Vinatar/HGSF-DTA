import pandas as pd
import matplotlib.pyplot as plt

# 加载CSV文件
 # 将此替换为你的CSV文件路径
file_path = "loss_log_without_graph_info.csv"

data = pd.read_csv(file_path)

# 确保CSV文件中有'Epoch'和'Loss'列
if 'Epoch' in data.columns and 'Loss' in data.columns:
    # 提取Epoch和Loss列
    epochs = data['Epoch']
    loss = data['Loss']

    # 绘制损失函数图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label='Loss', color='blue', linewidth=2)

    # 添加标题和标签
    #plt.title('Loss Function Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("CSV文件中缺少'Epoch'或'Loss'列")
