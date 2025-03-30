import argparse
import re, os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def get_words(filename):
    """
    读取文本文件，过滤无效字符并利用 jieba 分词，同时过滤长度为1的词
    """
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            tokens = list(cut(line))
            tokens = [token for token in tokens if len(token) > 1]
            words.extend(tokens)
    return words


def get_raw_text(filename):
    """
    直接读取文件的原始文本，用于TF-IDF向量化
    """
    with open(filename, 'r', encoding='utf-8') as fr:
        return fr.read()


def freq_pipeline(train_files):
    """
    高频词特征提取流程：
    1. 针对所有训练邮件分词
    2. 统计所有词汇频率，选取出现次数最多的100个词作为特征
    3. 对每封邮件构建包含每个特征词出现次数的向量
    """
    corpus_words = []
    for file in train_files:
        corpus_words.append(get_words(file))
    freq = Counter(chain(*corpus_words))
    top_words = [word for word, count in freq.most_common(100)]
    vectors = []
    for words in corpus_words:
        vector = [words.count(word) for word in top_words]
        vectors.append(vector)
    X = np.array(vectors)
    return X, top_words


def tfidf_pipeline(train_files):
    """
    TF-IDF特征提取流程：
    1. 针对所有训练邮件读取原始文本
    2. 使用 TfidfVectorizer（并结合自定义 jieba 分词器）将文本转换为TF-IDF特征矩阵
    """
    corpus = []
    for file in train_files:
        text = get_raw_text(file)
        corpus.append(text)

    def jieba_tokenizer(text):
        tokens = list(cut(text))
        return [token for token in tokens if len(token) > 1]

    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer, max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def main():
    parser = argparse.ArgumentParser(description="朴素贝叶斯邮件分类")
    parser.add_argument('--feature_method', type=str, default='freq', choices=['freq', 'tfidf'],
                        help='特征提取方式: freq表示高频词，tfidf表示TF-IDF加权')
    parser.add_argument('--balance', action='store_true', help='是否进行样本平衡处理 (SMOTE)')
    parser.add_argument('--evaluate', action='store_true', help='是否输出模型评估报告（划分测试集）')
    args = parser.parse_args()

    # 训练数据：0.txt到150.txt（共151封邮件）
    train_files = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    # 标签设置：0~126为垃圾邮件（标签1），127~150为普通邮件（标签0）
    labels = np.array([1] * 127 + [0] * 24)

    # 根据参数选择特征提取方式
    if args.feature_method == 'freq':
        X, top_words = freq_pipeline(train_files)
        print("使用高频词特征提取方式")
    else:
        X, vectorizer = tfidf_pipeline(train_files)
        print("使用TF-IDF加权特征提取方式")

    # 若请求评估，则划分出一部分数据作为测试集（此处采用20%比例，且保持类别分布均衡）
    if args.evaluate:
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
    else:
        X_train = X
        y_train = labels

    # 若请求样本平衡，则对训练数据使用SMOTE进行过采样处理
    if args.balance:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("进行了SMOTE样本平衡处理")

    # 训练多项式朴素贝叶斯模型
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # 若进行模型评估，则对测试集进行预测，并输出分类评估报告
    if args.evaluate:
        y_pred = model.predict(X_test)
        print("\n分类评估报告：")
        print(classification_report(y_test, y_pred))

    # 对未知邮件（151.txt～155.txt）进行分类预测
    test_files = ['邮件_files/{}.txt'.format(i) for i in range(151, 156)]
    for file in test_files:
        if args.feature_method == 'freq':
            words = get_words(file)
            vector = np.array([words.count(word) for word in top_words]).reshape(1, -1)
        else:
            text = get_raw_text(file)
            vector = vectorizer.transform([text])
        result = model.predict(vector)
        label = '垃圾邮件' if result[0] == 1 else '普通邮件'
        print('{} 分类情况: {}'.format(file, label))


if __name__ == "__main__":
    main()
