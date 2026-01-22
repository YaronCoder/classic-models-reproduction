import numpy as np
import matplotlib.pyplot as plt


# --- 1. 定义 Sigmoid 函数 ---
def sigmoid(x):
    """
    计算 Sigmoid 函数值: 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))

# --- 2. 准备数据 ---
# 生成从 -10 到 10 的一系列 x 值
x = np.linspace(-10, 10, 400)
# 计算对应的 sigmoid(x) 值
y = sigmoid(x)

# --- 3. 绘图 ---
plt.figure(figsize=(8, 6))

# 绘制主曲线
plt.plot(x, y, label='Sigmoid Function $\sigma(x) = \frac{1}{1+e^{-x}}$', color='blue', linewidth=2)

# 添加辅助线
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Threshold (0.5)') # 0.5 阈值线
plt.axhline(y=0, color='gray', linestyle=':', linewidth=1) # y=0 底线
plt.axhline(y=1, color='gray', linestyle=':', linewidth=1) # y=1 顶线
plt.axvline(x=0, color='gray', linestyle=':', linewidth=1) # x=0 中心线

# 添加图例、标题和标签
plt.title('Visualization of the Sigmoid Function (Binary Classification)')
plt.xlabel('Input Value (x) / Logits')
plt.ylabel('Output Probability (p)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([-10, 10])
plt.ylim([-0.1, 1.1]) # 稍微留点边距

# 显示图像
plt.show()