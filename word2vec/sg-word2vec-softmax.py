import numpy as np
import collections


class SoftmaxWord2Vec:
    """
    原始 Word2Vec 实现
    使用 Softmax 全词表多分类 (不使用负采样)
    """

    def __init__(self, vector_size=10, window=2, learning_rate=0.01):
        self.vector_size = vector_size
        self.window = window
        self.lr = learning_rate
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = 0
        self.W_in = None
        self.W_out = None

    def build_vocab(self, corpus):
        """与之前相同，构建词表"""
        words = []
        for sentence in corpus:
            words.extend(sentence.split())
        word_counts = collections.Counter(words)
        self.id2word = {i: word for i, word in enumerate(word_counts)}
        self.word2id = {word: i for i, word in self.id2word.items()}
        self.vocab_size = len(self.id2word)

        # 初始化权重
        self.W_in = np.random.uniform(-0.01, 0.01, (self.vocab_size, self.vector_size))
        self.W_out = np.random.uniform(-0.01, 0.01, (self.vocab_size, self.vector_size))
        print(f"Vocab size: {self.vocab_size}")

    def softmax(self, x):
        """
        计算整个向量的 Softmax
        x: shape [vocab_size]
        """
        # 减去最大值防止指数爆炸 (数值稳定性技巧)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def train(self, corpus, epochs=100):
        data = []
        for sentence in corpus:
            data.append([self.word2id[w] for w in sentence.split()])

        for epoch in range(epochs):
            loss = 0
            for sentence_ids in data:
                for i, center_id in enumerate(sentence_ids):

                    # 1. 获取中心词向量 u
                    u = self.W_in[center_id]

                    # 2. 确定上下文 (真实标签)
                    start = max(0, i - self.window)
                    end = min(len(sentence_ids), i + self.window + 1)
                    context_ids = sentence_ids[start:end]
                    context_ids = [idx for idx in context_ids if idx != center_id]

                    # 3. 遍历每一个上下文词 (标准多分类训练)
                    for target_id in context_ids:
                        # === 区别 A: 计算所有词的 Logits ===
                        # u (1, dim) dot W_out.T (dim, vocab_size) -> (1, vocab_size)
                        # 我们需要计算中心词和词表中"每一个词"的内积
                        logits = np.dot(u, self.W_out.T)

                        # === 区别 B: 使用 Softmax 计算概率分布 ===
                        y_hat = self.softmax(logits)

                        # === 区别 C: 计算误差 (y_hat - y) ===
                        # y 是 One-hot 向量，只有 target_id 位置是 1，其他是 0
                        # 所以误差向量 e 在 target_id 位置是 y_hat - 1，其他位置是 y_hat - 0
                        e = y_hat.copy()
                        e[target_id] -= 1  # 对应 label=1 的位置

                        # === 区别 D: 反向传播 (更新量极其巨大) ===
                        # 1. W_out 的梯度:
                        # 我们需要更新整个 W_out 矩阵，因为 softmax 让所有词都参与了计算
                        # dL/dW_out = e (vocab_size, 1) * u (1, dim) -> (vocab_size, dim)
                        grad_W_out = np.outer(e, u)

                        # 2. W_in 的梯度:
                        # dL/du = e (1, vocab_size) * W_out (vocab_size, dim) -> (1, dim)
                        grad_u = np.dot(e, self.W_out)

                        # 更新参数
                        self.W_out -= grad_W_out * self.lr
                        self.W_in[center_id] -= grad_u * self.lr

            if epoch % 100 == 0:
                print(f"Epoch {epoch} finished.")

    def similarity(self, word1, word2):
        """同上"""
        if word1 not in self.word2id or word2 not in self.word2id:
            return 0
        v1 = self.W_in[self.word2id[word1]]
        v2 = self.W_in[self.word2id[word2]]
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# 运行测试
corpus = [
    "he is a king", "she is a queen",
    "he is a man", "she is a woman",
    "king is powerful", "queen is beautiful"
]
w2v = SoftmaxWord2Vec(vector_size=8, window=2, learning_rate=0.05)
w2v.build_vocab(corpus)
w2v.train(corpus, epochs=500)

print("\n=== Softmax版本 测试 ===")
print(f"he vs king: {w2v.similarity('he', 'king'):.4f}")
print(f"he vs queen: {w2v.similarity('he', 'queen'):.4f}")
print(f"king vs queen: {w2v.similarity('king', 'queen'):.4f}")