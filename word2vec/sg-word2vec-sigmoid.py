import numpy as np
import collections


class Word2Vec:
    """
    Simple word2vec implementation.
    使用负采样
    """

    def __init__(self, vector_size=10, window=2, learning_rate=0.01):
        self.vector_size = vector_size  # 词向量维度
        self.window = window  # 窗口大小 (c)
        self.lr = learning_rate  # 学习率
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = 0

        # 两个矩阵：W_in (中心词向量), W_out (上下文/负样本向量)
        self.W_in = None
        self.W_out = None

    def build_vocab(self, corpus):
        """1. 数据预处理：构建词表"""
        words = []
        for sentence in corpus:
            words.extend(sentence.split())

        # 统计词频（这里为了简单没做去低频词）
        word_counts = collections.Counter(words)
        self.id2word = {i: word for i, word in enumerate(word_counts)}
        self.word2id = {word: i for i, word in self.id2word.items()}
        self.vocab_size = len(self.id2word)

        # 初始化权重 (随机小数)
        # W_in: [Vocab_Size, Vector_Size]
        # W_out: [Vocab_Size, Vector_Size]
        self.W_in = np.random.uniform(-0.01, 0.01, (self.vocab_size, self.vector_size))
        self.W_out = np.random.uniform(-0.01, 0.01, (self.vocab_size, self.vector_size))

        print(f"Vocab size: {self.vocab_size}")

    def sigmoid(self, x):
        # 防止溢出的安全sigmoid
        return 1 / (1 + np.exp(-np.clip(x, -6, 6)))

    def get_negative_samples(self, target_idx, context_ids, k=5):
        """负采样：随机抽取 k 个不等于 target_idx 的词，并且这个词不在当前的滑动窗口内"""
        neg_indices = []
        while len(neg_indices) < k:
            rand_idx = np.random.randint(0, self.vocab_size)
            if rand_idx != target_idx and rand_idx not in context_ids:
                neg_indices.append(rand_idx)
        return neg_indices

    def train(self, corpus, epochs=100):
        """训练主循环"""
        # 将语料转为 ID 列表
        data = []
        for sentence in corpus:
            data.append([self.word2id[w] for w in sentence.split()])

        for epoch in range(epochs):
            loss = 0
            for sentence_ids in data:
                # 遍历句子中的每个词作为中心词
                for i, center_id in enumerate(sentence_ids):

                    # === 1. 确定滑动窗口范围 ===
                    # 也就是 [i-window, i+window]，注意边界
                    start = max(0, i - self.window)
                    end = min(len(sentence_ids), i + self.window + 1)

                    context_ids = sentence_ids[start:end]
                    # 上下文不能包含中心词自己
                    context_ids = [idx for idx in context_ids if idx != center_id]

                    # 取出中心词向量 u (对应 W_in 的一行)
                    u = self.W_in[center_id]

                    # === 2. 遍历每一个上下文词 (正样本) ===
                    for context_id in context_ids:
                        # --- 正样本训练 ---
                        # 取出上下文向量 v_pos (对应 W_out 的一行)
                        v_pos = self.W_out[context_id]

                        # 前向传播: z = u · v
                        z = np.dot(u, v_pos)
                        pred = self.sigmoid(z)

                        # 梯度计算 (Sigmoid + CrossEntropy 的导数非常简洁: pred - label)
                        # 正样本 label = 1, 所以是 pred - 1
                        g_pos = (pred - 1) * self.lr

                        # 暂存 u 的梯度 (因为 u 还要参与负样本计算，不能马上更新 u)
                        u_grad = g_pos * v_pos

                        # 立即更新 v_pos (W_out)
                        self.W_out[context_id] -= g_pos * u

                        # --- 负样本训练 (Negative Sampling) ---
                        neg_ids = self.get_negative_samples(context_id, context_ids, k=5)
                        for neg_id in neg_ids:
                            v_neg = self.W_out[neg_id]

                            z_neg = np.dot(u, v_neg)
                            pred_neg = self.sigmoid(z_neg)

                            # 负样本 label = 0, 所以梯度是 pred - 0 = pred
                            g_neg = (pred_neg - 0) * self.lr

                            # 累加 u 的梯度
                            u_grad += g_neg * v_neg

                            # 立即更新 v_neg (W_out)
                            self.W_out[neg_id] -= g_neg * u

                        # --- 最后更新中心词 u (W_in) ---
                        self.W_in[center_id] -= u_grad

            if epoch % 100 == 0:
                print(f"Epoch {epoch} finished.")

    def similarity(self, word1, word2):
        """计算余弦相似度"""
        if word1 not in self.word2id or word2 not in self.word2id:
            return 0
        v1 = self.W_in[self.word2id[word1]]
        v2 = self.W_in[self.word2id[word2]]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ================= 运行测试 =================

# 1. 准备一小段语料 (关于国王和皇后的经典例子)
corpus = [
    "he is a king",
    "she is a queen",
    "he is a man",
    "she is a woman",
    "king is powerful",
    "queen is beautiful",
]

# 2. 初始化并训练
w2v = Word2Vec(vector_size=10, window=2, learning_rate=0.05)
w2v.build_vocab(corpus)
w2v.train(corpus, epochs=1000)

# 3. 验证结果
print("\n=== 相似度测试 ===")
print(f"he vs king: {w2v.similarity('he', 'king'):.4f}")
print(f"he vs queen: {w2v.similarity('he', 'queen'):.4f}")
print(f"woman vs queen: {w2v.similarity('woman', 'queen'):.4f}")
print(f"king vs queen: {w2v.similarity('king', 'queen'):.4f}")
