# -*- coding:utf-8 -*-

import numpy as np

from scipy.stats import multinomial
from scipy.stats import dirichlet


def cumulative_algorithm(p: list) -> int:

    p = np.array(p)

    for i in range(1, len(p)):
        p[i] = p[i-1] + p[i]

    u = np.random.uniform(0, 1) * p[-1]

    i = 0
    while i < len(p):
        if p[i] > u:
            break
        i += 1

    return i


sport_list = np.array(["basketball", "football", "tennis", "volleyball"])
policy_list = np.array(["China", "USA", "Japan", "trade"])
science_list = np.array(["paper", "academic", "university"])

test_doc = [sport_list, policy_list, science_list]


class LatentDirichletAllocation:

    def __init__(self):

        # 文档数量M 主题数量K 单词数量 V
        self.M, self.K, self.V = -1, -1, -1

        # 文档-主题的Dirichlet分布参数 主题-单词的Dirichlet分布参数
        self.alpha, self.beta = -1, -1

        # 文档-主题的分布矩阵 主题-单词的分布矩阵
        self.theta, self.phi = -1, -1

        # 通过word_id查询word 通过word查询word_id
        self.word_dict, self.word_inv_dict = {}, {}
        pass

    def build_word_dict(self, doc):
        tmp_w_list = []
        for doc_item in doc:
            tmp_w_list.extend(list(set(doc_item)))
        w_list = list(set(tmp_w_list))

        self.V = len(w_list)
        self.M = len(doc)

        self.word_dict = {i: w for i, w in enumerate(w_list)}
        self.word_inv_dict = {w: i for w, i in enumerate(w_list)}

    def generate(self, alpha, beta, n, k,):
        dirichlet()
        pass

    def collapsed_gibbs_sampling(self, doc, k: int, alpha, beta, iter_num):

        """ 创建计数矩阵和计数向量 """
        doc_topic_mat = np.zeros([self.M, self.K])  # 文本-话题频率矩阵
        topic_word_mat = np.zeros([self.K, self.V])  # 话题-单词频率矩阵

        doc_topic_vector = np.zeros([self.M])  # 文本-话题和 向量
        topic_word_vector = np.zeros([self.K])  # 话题-单词和 向量

        word_topic_mapping = np.zeros_like(doc)  # 记录每个文本中每个单词对应的主题

        z = np.array([self.M])

        """ 初始化计数矩阵和计数向量 
            对给定的所有文本的单词序列 每个位置上随机指派一个话题 整体构成所有文本的话题序列
        """
        for (doc_id, doc_item) in enumerate(doc):  # 遍历所有文本
            for (word_id, word) in enumerate(doc_item):  # 遍历所有单词
                # 查询word对应的word_index
                word_index = self.word_inv_dict[word]
                # 从均匀分布中抽样一个话题 得到topic_id
                topic_id = np.random.randint(0, self.K)
                # 记录当前文本对应的主题
                word_topic_mapping[doc_id, word_id] = topic_id
                # 根据采样出的话题，更新四个计数单元
                doc_topic_mat[doc_id, topic_id] += 1
                topic_word_mat[topic_id, word_index] += 1
                doc_topic_vector[doc_id] += 1
                topic_word_vector[topic_id] += 1

        """ 燃烧期准备 
            在每一个位置上计算在该位置上的话题的 满条件概率分布
            然后进行随机抽样 得到该位置的新的话题 分派给这个位置
        """
        for (doc_id, doc_item) in enumerate(doc):  # 遍历所有文本
            for (word_id, word) in enumerate(doc_item):  # 遍历所有单词

                # 查询word对应的word_index和topic_id
                word_index = self.word_inv_dict[word]
                topic_id = word_topic_mapping[doc_id, word_id]
                # 根据当前单词对应的话题，更新四个计数单元
                doc_topic_mat[doc_id, topic_id] -= 1
                topic_word_mat[topic_id, word_index] -= 1
                doc_topic_vector[doc_id] -= 1
                topic_word_vector[topic_id] -= 1

                # 按照 满条件分布 进行抽样
                topic_prob_list = self.full_condition_dist()

    def full_condition_dist(self) -> list:
        """ 按照满条件分布进行抽样 """
        pass


if __name__ == '__main__':
    pass