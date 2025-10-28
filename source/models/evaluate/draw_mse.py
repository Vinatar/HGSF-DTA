import pandas as pd
import matplotlib.pyplot as plt

# 加载CSV文件
 # 将此替换为你的CSV文件路径
file_path = "fold0.csv"

data = pd.read_csv(file_path)

# 确保CSV文件中有'Epoch'和'MSE'列
if 'Epoch' in data.columns and 'MSE' in data.columns:
    # 提取Epoch和Loss列
    epochs = data['Epoch']
    loss = data['MSE']

    # 绘制MSE图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label='MSE', color='blue', linewidth=2)

    # 添加标题和标签
    #plt.title('MSE Function Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("CSV文件中缺少'Epoch'或'MSE'列")
