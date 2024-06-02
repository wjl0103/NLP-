import os
import re
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

# 路径设置
corpus_dir = 'yuliaoku'
stopwords_path = os.path.join(corpus_dir, 'cn_stopwords.txt')
inf_path = os.path.join(corpus_dir, 'inf.txt')

# 读取小说文件名
with open(inf_path, 'r', encoding='gbk', errors='ignore') as f:
    novel_files = f.read().strip().split(',')

# 读取停用词
with open(stopwords_path, 'r', encoding='gbk', errors='ignore') as f:
    stopwords = set(f.read().strip().split('\n'))

# 预处理函数
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # 移除多余空白
    words = jieba.lcut(text)  # 分词
    words = [word for word in words if word not in stopwords and re.match(r'\w', word)]  # 去停用词和非中文字符
    return words

# 读取并预处理所有小说
corpus = []
for novel_file in novel_files:
    with open(os.path.join(corpus_dir, novel_file + '.txt'), 'r', encoding='gbk', errors='ignore') as f:
        text = f.read().replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com\n', '')
        words = preprocess_text(text)
        corpus.append(words)

# 训练Word2Vec模型
model = Word2Vec(sentences=corpus, vector_size=100, window=5, sg=1, min_count=5, workers=4)

# 保存模型
model.save("word2vec_jinyong.model")

# 加载模型
model = Word2Vec.load("word2vec_jinyong.model")

# 计算词向量之间的语义距离
def word_similarity(word1, word2):
    return model.wv.similarity(word1, word2)

similarity = word_similarity('郭靖', '黄蓉')
print(f"郭靖和黄蓉的相似度: {similarity}")
similarity_1 = word_similarity('郭靖', '韦小宝')
print(f"郭靖和韦小宝的相似度: {similarity_1}")

# 获取词向量
words = list(model.wv.index_to_key)
word_vectors = np.array([model.wv[word] for word in words])

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=10, random_state=0).fit(word_vectors)

# 输出每个簇的词语
clusters = {i: [] for i in range(10)}
for word, cluster in zip(words, kmeans.labels_):
    clusters[cluster].append(word)

for cluster, words in clusters.items():
    print(f"簇 {cluster}: {words[:10]}")  # 每个簇打印前10个词语