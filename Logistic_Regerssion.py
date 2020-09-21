import numpy as np

def loaddatas():
    """
    x的维度（特征数，样本数）
    y的维度（1，样本数）
    """
    # 原始数据维度（样本数，特征数），（17，2）
    x_origin = np.array([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211],
                         [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042], [0.719,0.103]])
    # 原始标签维度(样本数，1)，（17，1）
    y_origin = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]])

    # 将数据集转置，维度变为（特征数，样本数），（2，17）
    x = x_origin.T
    # 将标签机转置，维度变为（1，样本数）
    y = y_origin.T

    print(x.shape)
    print(y.shape)

    return x, y


# 对率回归
def logit_regression(x, y, examples):
    # 初始化差值deta_l
    deta_l = 0.1
    # 循环次数
    epoch = 0
    # 初始化对数似然
    cur_l = 0
    # 初始化W和b
    # w是个列向量，维度和数据x的维度一样 w维度为（特征数，1）, w维度：（2，1）
    w = np.ones((x.shape[0], 1))
    # b是一个一维的列向量，                                 b维度：（1，1）
    b = np.ones((1,1))
    # β = (w;b) β维度为（特征数+1，1），                    β维度：（3，1）
    beta = np.r_[w,b]
    # x^ = (x;1) x^维度为（特征数+1，样本数）               x^维度：（3，17）
    x_hat = np.r_[x, np.ones((1, x.shape[1]))]                   
    
    # 当两次的β之间的差值小于0.000001可认为收敛
    while(abs(deta_l) > 0.000001):
        epoch = epoch + 1

        # 先验概率  维度（1，17）
        prior_p = np.dot(beta.T, x_hat)/(1 + np.dot(beta.T, x_hat))
        # 后验概率  维度（1，17）
        posterior_p = 1/(1 + np.dot(beta.T, x_hat))
        # 把上一次对数似然值赋给old_l
        old_l = cur_l               
        # np.dot(beta.T, x_hat) 维度（1，17）
        # y维度（1，17），np.multiply(y,np.dot(beta.T, x_hat))对应位置元素相乘，结果维度仍为（1，17）
        # np.log(1 + np.exp(np.dot(beta.T, x_hat))) 对每个元素进行指数运算再加1，再对数运算，结果维度为（1，17）
        # np.sum将17个元素累加
        cur_l = np.sum(-1 * np.multiply(y,np.dot(beta.T, x_hat)) + np.log(1 + np.exp(np.dot(beta.T, x_hat))))
        
        # 牛顿法迭代更新

        # (y - prior_p) 维度（1，17）
        # 根据公式，x^中每个样本的属性值（也就是每一列）乘以y的每一列的元素，也就是对应位置元素相乘。使用np.multiply，同时会使用广播机制，将y扩充成（3，17）
        # np.multiply(x_hat,(y - prior_p).T) 维度（3，17）
        # 根据公式，d_beta 需要把所有样本的这个计算结果累加，也就是按列累加，因此采用np.sum，并定义axis=1
        # 通过reshape确保生成的向量维度是正确的
        # d_beta 维度（3，1）
        d_beta = -np.sum(np.multiply(x_hat,(y - prior_p)), axis=1).reshape(3,1)
        # x_plus维度（3，3）
        x_plus = np.dot(x_hat, x_hat.T)
        # np.multiply(prior_p, posterior_p)维度（1，17）
        p_p = np.multiply(prior_p, posterior_p)
        # d2_beta 维度（3，3）
        d2_beta = np.zeros((x_hat.shape[0], x_hat.shape[0]))
        for i in range(examples):
            # x[:,i].reshape(3,1) 取x_hat的第i列元素，通过reshape保证是一个列向量
            # x[:,i].reshape(3,1) 取x_hat的第i列元素的装置，通过reshape保证是一个行向量
            d2_beta = d2_beta + np.dot(x_hat[:,i].reshape(3,1), x_hat[:,i].reshape(1,3))
        # 更新
        # d2_beta.I计算逆矩阵
        # beta维度（3，1）
        beta = beta - np.dot(np.matrix(d2_beta).I, d_beta)
        # 计算前后两次似然差值
        deta_l = cur_l - old_l

        print(str(epoch) + "    " + str(deta_l) +  "    "  + str(beta))
    
    print(deta_l)
    print("总共迭代了：" + str(epoch))
    print("beta: " + str(beta))



# 加载数据
data, target = loaddatas()
# 迭代训练
logit_regression(data, target, data.shape[1])


