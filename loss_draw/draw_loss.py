# draw_loss.py

"""
这个文件用于绘制损失函数图像。
"""

# 导入必要的模块
import matplotlib.pyplot as plt
import numpy as np
import re

LOG_RE = r"Epoch (\d+/\d+).*?Iter: (\d+/\d+).*?lr\(PiecewiseDecay\): ([\d.]+).*?top1: ([\d.]+).*?top5: ([\d.]+).*?CELoss: ([\d.]+).*?loss: ([\d.]+).*?batch_cost: ([\d.]+)s.*?ips: ([\d.]+).*?eta: (\d+:\d+:\d+)"
LOSS_RE = r"loss: ([\d.]+)"
# 如果需要其他模块，可以在这里继续导入

def check_string(re_exp, str):
    res = re.search(re_exp, str)
    if res:
        return True
    else:
        return False

def Read_file(file_path):
    """
    读取文件内容。
    
    参数:
        file_path (str): 文件路径。
    
    返回:
        list or numpy.array: 文件中的数据列表或数组。
    """
    try:
        loss = []
        with open(file_path, 'r') as f:
            for line in f:
                if(check_string(LOG_RE, line)):
                    loss_data = re.search(LOSS_RE, line).group(1)
                    loss.append(float(loss_data))
            return loss
    except Exception as e:
        raise e

def draw_loss_function(losses):
    """
    绘制损失函数图像。
    
    参数:
        losses (list or numpy.array): 损失值列表。
    """
    # 在这里添加绘制损失函数的代码，但不要写实现细节
    y_train_loss = losses
    x_train_loss = np.arange(len(y_train_loss))
    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    # plt.scatter(x_train_loss, y_train_loss, color='blue', s=5)
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle='solid', label='Loss')
    plt.legend()
    plt.title('Paddle ResNet50 XPU Loss')
    plt.savefig('loss.png')

def main():
    """
    主程序入口。
    """
    # 在这里添加调用 draw_loss_function 的代码，以及可能的其他逻辑
    loss = Read_file('train_256.log')
    # loss = loss[:250]
    draw_loss_function(loss)



if __name__ == "__main__":
    main()