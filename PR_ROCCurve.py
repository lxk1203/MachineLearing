import numpy as np
import matplotlib.pyplot as plt

TRUE_EXAMPLES = 25
FALSE_EXAMPLES = 25


def create_data(true_examples, false_example):
    # 创建真值数组和预测值数组
    # 创建两个0,1数组进行拼接并打乱
    y_true = np.concatenate((np.zeros(true_examples), np.ones(false_example)))
    np.random.shuffle(y_true)
    # 创建得分数组
    y_score = np.random.rand(true_examples + false_example)
    # argsort可以获取排序后的值对应原来的索引，默认升序，通过[::-1]实现转置，变成降序  
    y_score_index = np.argsort(y_score)[::-1]
    # sort对得分数组进行排序，通过[::-1]实现转置，变成降序，用于后面的ROC作图
    y_score = np.sort(y_score)[::-1]

    return y_true, y_score, y_score_index


def create_coordinates(t_true, t_score, t_score_index):
    # 初始化横纵坐标
    x = 0
    y = 0
    # 初始化横纵坐标集合，用于作图
    c_x = []
    c_y = []
    # (0,0)是第一个点
    c_x.append(x)
    c_y.append(y)
    # 预测值
    pred = 1
    for i in range(len(t_score)):           # 遍历降序排列的得分数组
        # 通过t_score_index获取得分对应的原索引，这个索引也是该得分对应真值的索引（真值数组和未排序得分数组元素一一对应），通过索引获取真值
        gtrue = t_true[t_score_index[i]]
        if pred == gtrue:                  # 为真阳性
            x = x
            y = y + 1/TRUE_EXAMPLES
        else:                               # 为假阳性
            x = x + 1/FALSE_EXAMPLES
            y = y
        # 添加新坐标
        c_x.append(x)
        c_y.append(y)

    return c_x, c_y


# 画图函数
def draw_curve(c_x, c_y):
    plt.plot(c_x, c_y)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("RPC curve")
    plt.show()


test_true, test_score, test_score_index = create_data(TRUE_EXAMPLES, FALSE_EXAMPLES)
coordinate_x, coordinate_y = create_coordinates(test_true, test_score, test_score_index)
draw_curve(coordinate_x, coordinate_y)



