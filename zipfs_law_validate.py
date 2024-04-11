import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import jieba
import numpy as np

def load_text(file_path):
    """从文件中加载文本数据。"""
    with open(file_path, 'r', encoding='gbk',errors='ignore') as file:
        return file.read().replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com\n', '')


def load_stopwords(file_path):
    """从文件中加载停词列表。"""
    with open(file_path, 'r', encoding='gbk',errors='ignore') as file:
        return set(file.read().splitlines())


def preprocess_text(text, stopwords):
    """预处理文本数据: 分词并删除标点符号和停词。"""
    words = jieba.cut(text)
    words = [word for word in words if word.strip() and word not in stopwords]
    return words


def plot_zipfs_law(word_freq):
    """绘制词频分布图以验证Zipf定律。"""
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:1000]
    ranks = range(1, len(sorted_word_freq) + 1)
    freqs = [freq for _, freq in sorted_word_freq]

    plt.figure(figsize=(10, 6))
    plt.plot(np.log(ranks), np.log(freqs), marker='o', linestyle='-')
    plt.xlabel('Log Rank')
    plt.ylabel('Log Frequency')
    plt.title('Zipf\'s Law Verification (Top 1000)')
    plt.grid(True)
    plt.show()


def main():
    # 加载停词列表
    stopwords_file_path = os.path.join('中文语料', 'cn_stopwords.txt')  # 停词文件路径
    stopwords = load_stopwords(stopwords_file_path)

    # 加载小说名称列表
    inf_file_path = os.path.join('中文语料', 'inf.txt')  # 小说名称文件路径
    novel_names = load_text(inf_file_path).split(',')

    # 加载小说全文
    novels_folder = '中文语料'  # 小说文件夹路径
    all_texts = ''
    for novel_name in novel_names:
        novel_file_path = os.path.join(novels_folder, novel_name.strip() + '.txt')
        if os.path.exists(novel_file_path):
            all_texts += load_text(novel_file_path)

    # 预处理文本并统计词频
    words = preprocess_text(all_texts, stopwords)
    word_freq = Counter(words)

    # 绘制词频分布图
    plot_zipfs_law(word_freq)


if __name__ == "__main__":
    main()
