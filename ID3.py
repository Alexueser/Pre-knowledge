import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random


# 计算信息熵的函数
def entropy(y):
    hist = np.bincount(y)  # 计算各个类别的数量
    ps = hist / len(y)  # 计算各个类别的概率
    return -np.sum([p * np.log2(p) for p in ps if p > 0])  # 计算信息熵


# 定义决策树的节点类
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # 分裂特征的索引
        self.threshold = threshold  # 分裂特征的阈值
        self.left = left  # 左子节点
        self.right = right  # 右子节点
        self.value = value  # 当前节点的类别值（仅在叶子节点有效）

    def is_leaf_node(self):
        return self.value is not None  # 如果节点的值不为空，说明是叶子节点


# ID3算法实现
def id3(X, y, min_samples_split=2, max_depth=5, depth=0):
    n_samples, n_features = X.shape

    # 如果当前节点的样本数大于等于最小分裂样本数，且深度小于等于最大深度，则进行分裂
    if n_samples >= min_samples_split and depth <= max_depth:
        feature, threshold = find_best_split(X, y, n_features)  # 找到最佳分裂特征和阈值
        if feature is not None:
            indices_left = X[:, feature] <= threshold  # 根据阈值划分左子节点样本
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]
            return Node(
                feature=feature,
                threshold=threshold,
                left=id3(X_left, y_left, min_samples_split, max_depth, depth + 1),
                right=id3(X_right, y_right, min_samples_split, max_depth, depth + 1),
            )

    # 当满足终止条件时，返回叶子节点
    return Node(value=Counter(y).most_common(1)[0][0])


# 寻找最佳分裂特征和阈值的函数
def find_best_split(X, y, n_features):
    best_gain = -1  # 初始化最佳信息增益
    best_feature, best_threshold = None, None  # 初始化最佳特征和阈值
    root_entropy = entropy(y)  # 计算当前节点的信息熵

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])  # 获取当前特征的所有可能阈值
        for threshold in thresholds:
            indices_left = X[:, feature] <= threshold  # 根据阈值划分左子节点样本
            y_left = y[indices_left]
            y_right = y[~indices_left]
            if len(y_left) == 0 or len(y_right == 0):
                continue

            p_left, p_right = len(y_left) / len(y), len(y_right) / len(y)  # 计算左右子节点在总样本中的比例
            current_entropy = p_left * entropy(y_left) + p_right * entropy(y_right)  # 计算当前特征和阈值的信息熵
            gain = root_entropy - current_entropy  # 计算信息增益

            if gain > best_gain:  # 如果当前信息增益大于最佳信息增益，则更新最佳特征和阈值
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


# 预测单个样本的函数
def predict_sample(node, x):
    if node.is_leaf_node():  # 如果当前节点是叶子节点，则返回节点的类别值
        return node.value

    if x[node.feature] <= node.threshold:  # 如果样本的特征值小于等于阈值，则进入左子节点
        return predict_sample(node.left, x)
    return predict_sample(node.right, x)  # 否则进入右子节点


# 预测一组样本的函数
def predict(tree, X):
    return np.array([predict_sample(tree, x) for x in X])


# 随机生成数据
np.random.seed(42)
X = np.random.rand(200, 2)
y = np.array([1 if x[0] > x[1] else 0 for x in X])
y = np.array([random.choice([0, 1]) if random.random() < 0.1 else y_i for y_i in y])  # 添加一些噪声

# 训练ID3决策树
tree = id3(X, y)


# 可视化数据的函数
def plot_data(X, y, tree):
    plt.figure(figsize=(10, 7))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='Class 0')  # 画出类别0的散点图
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Class 1')  # 画出类别1的散点图

    x0_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xx, yy = np.meshgrid(x0_range, x1_range)  # 生成网格数据
    grid = np.c_[xx.ravel(), yy.ravel()]  # 将网格数据转换为表格数据

    predictions = predict(tree, grid)  # 预测网格数据的类别
    zz = predictions.reshape(xx.shape)  # 将预测结果转换为网格数据

    plt.contourf(xx, yy, zz, alpha=0.2, cmap='bwr')  # 画出决策边界
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


plot_data(X, y, tree)
