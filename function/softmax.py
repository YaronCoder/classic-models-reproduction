import numpy as np
import matplotlib.pyplot as plt

# --- 1. 定义 Softmax 函数 ---
def softmax(z):
    """
    计算 Softmax 函数。
    输入 z 是一个二维数组 (N_samples, N_classes)。
    """
    # 为了数值稳定性，通常会减去最大值
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# --- 2. 准备数据 ---
# 假设有3个类别。我们固定前两个类别的logits，改变第三个类别的logit。
# Class 0 logit = 2.0
# Class 1 logit = 1.0
# Class 2 logit = 从 -3 变化到 5

z2_values = np.linspace(-3, 5, 400)
N = len(z2_values)

# 构建输入的 logits 矩阵，形状为 (400, 3)
# 每一行是一个样本的 [z0, z1, z2]
z_matrix = np.zeros((N, 3))
z_matrix[:, 0] = 2.0  # 固定 z0
z_matrix[:, 1] = 1.0  # 固定 z1
z_matrix[:, 2] = z2_values # 变化的 z2

# 计算 Softmax 概率
# probabilities 的形状也是 (400, 3)
probabilities = softmax(z_matrix)

# --- 3. 绘图 ---
plt.figure(figsize=(10, 6))

# 分别绘制三个类别的概率变化曲线
plt.plot(z2_values, probabilities[:, 0], label='P(Class 0) | fixed z0=2.0', linewidth=2)
plt.plot(z2_values, probabilities[:, 1], label='P(Class 1) | fixed z1=1.0', linewidth=2)
plt.plot(z2_values, probabilities[:, 2], label='P(Class 2) | varying z2', linewidth=3, linestyle='--')

# 添加图例、标题和标签
plt.title('Dynamics of Softmax Probabilities (Competition between Classes)')
plt.xlabel('Logit value of Class 2 (z2)')
plt.ylabel('Output Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([-0.05, 1.05])

# 添加标注点
# 当 z2 = 2.0 时，z = [2.0, 1.0, 2.0]。此时 P(Class 0) 应该等于 P(Class 2)
idx = np.abs(z2_values - 2.0).argmin()
plt.scatter([2.0, 2.0], [probabilities[idx, 0], probabilities[idx, 2]], color='red', zorder=5)
plt.text(2.1, probabilities[idx, 0]+0.05, 'z0=z2, Probabilities Equal', color='red')


# 显示图像
plt.show()