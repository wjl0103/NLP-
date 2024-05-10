import os
import random
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def load_stopwords(file_path):
    """从文件中加载停词列表。"""
    with open(file_path, 'r', encoding='gbk', errors='ignore') as file:
        return set(file.read().splitlines())

# 读取停词列表
stopwords = load_stopwords(os.path.join("yuliaoku", "cn_stopwords.txt"))

# 读取小说名字
novels_file = "yuliaoku/inf.txt"
with open(novels_file, 'r', encoding='gbk', errors='ignore') as f:
    novels = f.read().split(',')

# 创建小说到标签的映射
novel_to_label = {novel: i + 1 for i, novel in enumerate(novels)}

# 读取所有小说的段落，并为每个段落分配正确的标签
all_paragraphs = []
all_labels = []
for novel in novels:
    with open(os.path.join("yuliaoku", novel + ".txt"), 'r', encoding='gbk', errors='ignore') as f:
        content = f.read().replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com\n', '')  # 删除无用文字
        paragraphs = re.split(r'\n\u3000\u3000', content)  # 使用正则表达式按照全角空格和换行符分割段落
        # 过滤停词
        paragraphs = [' '.join([word for word in jieba.cut(paragraph) if word not in stopwords]) for paragraph in paragraphs]
        all_paragraphs.extend(paragraphs)
        all_labels.extend([novel_to_label[novel]] * len(paragraphs))

# 定义一个CountVectorizer对象
word_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)

# 将段落转换为文档-词频矩阵
X = word_vectorizer.fit_transform(all_paragraphs)

# 定义不同主题个数 T 下的交叉验证性能（以词为基本单位）
T_values = [5, 10, 15, 20]
results_T_word = []

for T in T_values:
    # 定义并拟合 LDA 模型
    lda = LatentDirichletAllocation(n_components=T, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(X)

    # 提取段落的主题分布
    paragraph_topic_distribution = lda.transform(X)

    # 定义随机森林分类器
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    # 进行交叉验证
    scores = cross_val_score(classifier, paragraph_topic_distribution, all_labels, cv=10)
    results_T_word.append(np.mean(scores))

print("以词为基本单位时，不同主题个数 T 下的分类性能：", results_T_word)

# 定义不同 token 数目 K 下的交叉验证性能（以词为基本单位）
K_values = [20, 100, 500, 1000, 3000]
results_K_word = []

for K in K_values:
    # 从每个小说中抽取段落作为数据集
    sampled_data = random.sample(list(zip(all_paragraphs, all_labels)), 1000)
    sampled_paragraphs, labels = zip(*sampled_data)

    # 将每个段落的 token 数目固定为 K
    sampled_paragraphs_K = [' '.join(paragraph.split()[:K]) for paragraph in sampled_paragraphs]

    # 将段落转换为文档-词频矩阵
    X_K = word_vectorizer.fit_transform(sampled_paragraphs_K)

    # 定义并拟合 LDA 模型
    lda = LatentDirichletAllocation(n_components=10, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(X_K)

    # 提取段落的主题分布
    paragraph_topic_distribution_K = lda.transform(X_K)

    # 定义随机森林分类器
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    # 进行交叉验证
    scores = cross_val_score(classifier, paragraph_topic_distribution_K, labels, cv=10)
    results_K_word.append(np.mean(scores))

print("以词为基本单位时，不同 token 数目 K 下的分类性能：", results_K_word)

# 重新定义 CountVectorizer 对象，以字为基本单元
char_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, analyzer='char', tokenizer=lambda x: x)

# 将段落转换为文档-词频矩阵（以字为基本单元）
X_char = char_vectorizer.fit_transform(all_paragraphs)

# 定义不同主题个数 T 下的交叉验证性能（以字为基本单位）
results_T_char = []

for T in T_values:
    # 定义并拟合 LDA 模型（以字为基本单位）
    lda_char = LatentDirichletAllocation(n_components=T, max_iter=5,
                                         learning_method='online',
                                         learning_offset=50.,
                                         random_state=0)
    lda_char.fit(X_char)

    # 提取段落的主题分布（以字为基本单位）
    paragraph_topic_distribution_char = lda_char.transform(X_char)

    # 定义随机森林分类器
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    # 进行交叉验证
    scores = cross_val_score(classifier, paragraph_topic_distribution_char, all_labels, cv=10)
    results_T_char.append(np.mean(scores))

print("以字为基本单位时，不同主题个数 T 下的分类性能：", results_T_char)
