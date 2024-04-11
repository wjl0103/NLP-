import os
import math
import jieba
from collections import Counter

def load_text(file_path):
    """从文件中加载文本数据。"""
    with open(file_path, 'r', encoding='gbk',errors='ignore') as file:
        return file.read().replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com\n', '')

def load_stopwords(file_path):
    """从文件中加载停词列表。"""
    with open(file_path, 'r', encoding='gbk',errors='ignore') as file:
        return set(file.read().splitlines())

def preprocess_text(text, stopwords):
    """预处理文本数据: 分词并删除停词。"""
    words = jieba.cut(text)
    words = [word for word in words if word.strip() and word not in stopwords]
    return words

def calculate_entropy(data):
    """计算给定数据的信息熵。"""
    total = len(data)
    counts = Counter(data)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy



def main():
    # 加载停词列表
    stopwords_file_path = '中文语料/cn_stopwords.txt'  # 停词文件路径
    stopwords = load_stopwords(stopwords_file_path)

    # 加载小说名称列表
    inf_file_path = '中文语料/inf.txt'  # 小说名称文件路径
    novel_names = load_text(inf_file_path).split(',')

    # 加载小说全文并计算信息熵
    novels_folder = '中文语料'  # 小说文件夹路径
    char_entropies = []
    word_entropies = []
    for novel_name in novel_names:
        novel_file_path = os.path.join(novels_folder, novel_name.strip() + '.txt')
        if os.path.exists(novel_file_path):
            novel_text = load_text(novel_file_path)
            words = preprocess_text(novel_text, stopwords)
            characters = list(novel_text)
            if len(characters) > 0:
                char_entropy = calculate_entropy(characters)
                char_entropies.append(char_entropy)
            if len(words) > 0:
                word_entropy = calculate_entropy(words)
                word_entropies.append(word_entropy)

    # 打印表格
    print("小说名称\t\t\t\t\t\t字单位信息熵\t\t词单位信息熵")
    for name, char_entropy, word_entropy in zip(novel_names, char_entropies, word_entropies):
        print(f"{name.strip():<30}\t{char_entropy:.4f}\t\t\t{word_entropy:.4f}")



if __name__ == "__main__":
    main()
