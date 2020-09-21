import math
import numpy as np
from collections import Counter

def load_datas():
    """
    训练集D{(x1,y1)...(xm,ym)}
        字符串二维数组
        维度：m * d+1, d 为数据的属性值，+1表示加上标签，m为样本总数
        每一行是一个样本
        每一列是一个属性
        最后一列的是标签y
    属性集A{a1,...,ad}
        是一个列表，存放每个属性的属性值列表
        列表索引就代表属性，例如：索引0表示"色泽"，索引1表示"根蒂"，...
        属性值列表存放属性值，以数字0,1,2...代表每个属性里的属性值
        维度：d*？，d表示数据有多少个属性，?表示该属性具体的属性值
    :return:
    """
    D = np.array([['qinglve', 'quansuo', 'zhuoxiang', 'qingxi', 'aoxian', 'yinghua', 1],
                 ['wuhei', 'quansuo', 'chenmen', 'qingxi', 'aoxian','yinghua', 1],
                 ['wuhei', 'quansuo', 'zhuoxiang', 'qingxi','aoxian','yinghua', 1],
                 ['qinglve', 'quansuo', 'chenmen', 'qingxi', 'aoxian','yinghua', 1],
                 ['qianbai', 'quansuo', 'zhuoxiang', 'qingxi','aoxian', 'yinghua', 1],
                 ['qinglve', 'shaoquan', 'zhuoxiang', 'qingxi', 'shaoao','ruannian', 1],
                 ['wuhei', 'shaoquan', 'zhuoxiang', 'shaohu', 'shaoao','ruannian', 1],
                 ['wuhei', 'shaoquan', 'zhuoxiang', 'qingxi', 'shaoao','yinghua', 1],
                 ['wuhei', 'shaoquan', 'chenmen', 'shaohu','shaoao','yinghua', 0],
                 ['qinglve', 'yingting', 'qingcui', 'qingxi', 'pingtan', 'ruannian', 0],
                 ['qianbai', 'yingting', 'qingcui', 'mohu', 'pingtan', 'yinghua', 0],
                 ['qianbai', 'quansuo', 'zhuoxiang', 'mohu', 'pingtan', 'ruannian', 0],
                 ['qinglve', 'shaoquan', 'zhuoxiang', 'shaohu', 'aoxian', 'yinghua', 0],
                 ['qianbai', 'shaoquan', 'chenmen', 'shaohu', 'aoxian', 'yinghua', 0],
                 ['wuhei', 'shaoquan', 'zhuoxiang', 'qingxi', 'shaoao', 'ruannian', 0],
                 ['qianbai', 'quansuo', 'zhuoxiang', 'mohu', 'pingtan', 'yinghua', 0],
                 ['qinglve', 'quansuo', 'chenmen', 'shaohu', 'shaoao', 'yinghua', 0]],
                 dtype=object)
    A = [['qinglve','wuhei','qianbai'],
         ['quansuo','shaoquan','yingting'],
         ['zhuoxiang','qingcui','chenmen'],
         ['qingxi','shaohu','mohu'],
         ['aoxian','shaoao','pingtan'],
         ['yinghua','ruannian']]

    return D, A

def allElementsEqual(ds, dim=1):
    """
    判断给定数组是否全部元素都相等
    :param ds：需要判断的标签数组
    :param dim：数组维度，默认一维
    :return: all_equal：True表示全部相等，False表示不全等
    """
    all_equal = True
    if dim == 1:
        if ds.size != 1:
            # 一维数组且元素不只有一个
            # 一维数组，.size获取数组元素个数
            for i in range(ds.size-1):
                if ds[i+1] != ds[i]:
                    all_equal = False
                    break
    elif dim == 2:
        # 二维数组，获取每一列属性值，比较是否相同
        for i in range(ds.shape[1]):
            all_equal = all_equal and allElementsEqual(ds[:,i])

    return all_equal


def ent_check(p):
    """
    对信息熵计算进行调整
        如果概率p=0，那么 plog2p=0，避免计算log2(0)出现错误
        否则进行正常计算 plog2p
    :param p: 样本比例
    :return: 信息熵Ent
    """
    if p == 0:
        return 0
    else:
        return -p*math.log(p, 2)

def ent(i_datas):
    """
    计算传递进来具体属性值的信息熵 Ent(D)=-Σpk*log2pk
    :param i_datas: 当前数据集
    :return: 属性值的信息熵
    """
    # 获取数据集的行，就是当前数据集的总样本数
    total_example = i_datas.shape[0]
    # np.where(i_datas[:,-1]==1) 返回i_datas标签当中等于1的索引，i_datas[:,-1]返回的是标签这一列的一维数组，符合条件的索引刚好就是在原二维数组对应的行索引
    # .shape[0] 获取行总数，也就是正例样本总数
    if total_example == 0:
        # 如果当前子集为空，也就不存在后面的正例比例（p_p）和负例比例（n_p)，因此直接认为Ent=0
        Ent = 0
    else:
        positive_examples = i_datas[np.where(i_datas[:,-1]==1)].shape[0]
        print("\tpositive_examples:" + str(positive_examples))
        # 对负例样本进行同样的操作
        negative_examples = i_datas[np.where(i_datas[:,-1]==0)].shape[0]
        print("\tnegative_examples:" + str(negative_examples))
        # 计算正例比例p_p
        p_p = positive_examples/total_example
        print("\tp_p:" + str(p_p))
        # 计算负例比例n_p
        n_p = negative_examples/total_example
        print("\tn_p:" + str(n_p))
        # 计算信息熵
        Ent = ent_check(p_p) + ent_check(n_p)
        print("\tEnt:" + str(Ent))

    return Ent


def bestAttribute(D, A):
    """
    根据数据集和属性集找出属性集中最优划分属性a*
    :param D: 当前数据集
    :param A: 当前属性集
    :return: 最优划分属性a*

    """
    """
    # datas[:,:-1] 将标签去掉，.flatten()将数据集展平为一维，用于后面计数
    # Counter对展平后的数据集里的每个数值进行计数；返回对象可以当做一个字典：key是元素，value是对应元素出现的次数
    Dvs = Counter(datas[:,:-1].flatten())
    # np.where(datas[:,-1]==1) 返回datas标签当中等于1的索引，datas[:,-1]返回的是标签这一列的一维数组，符合条件的索引刚好就是在原二维数组对应的行索引
    # Counter对标签为1的数据里的每个数值进行计数；
    Positives = Counter(datas[np.where(datas[:,-1]==1)])
    # 对标签为0的数据作为相同处理
    Negatives = Counter(datas[np.where(datas[:,-1]==0)])
    """
    # 获取总数据集D的样本数，用于后面的增益计算;行数即为总样本数
    D_examples = D.shape[0]
    # 计算总数据集D信息熵Ent(D)
    print("root:")
    Ent_D = ent(D)
    # 创建一个增益列表存放每个属性的信息增益
    gainList = []

    # 遍历整个属性集合，取出每个属性的索引和相应的属性值列表
    for index, attribute in enumerate(A):
        # 每计算一个属性的信息增益，初始化信息增益G=Ent(D)
        Gain = Ent_D
        # 遍历属性值列表，取出每个属性值
        for value in attribute:
            print(value + ":")
            # D[:,index] 属性在属性列表的索引值就是在数据集的列数，可以通过index对应起来
            # np.where(D[:,index]==value) 找出数据集中value属性值对应的行索引
            # D[np.where(D[:,index]==value)] 生成value属性值的子数据集
            Dv = D[np.where(D[:,index]==value)]
            # 获取子数据集Dv的样本数，用于后面的增益计算;行数即为总样本数
            Dv_examples = Dv.shape[0]
            # 信息增益 G(D,a)=Ent(D) - Σ|Dv|/|D|*Ent(Dv)
            Gain -= Dv_examples/D_examples*ent(Dv)
        print("----------------------")
        print(str(index) + ": " + str(Gain))
        print("----------------------")
        # 将计算得到的Gain加入增益列表
        gainList.append(Gain)

    # 获取最优划分属性a*，返回信息增益列表中最大值索引
    # 先获取增益最大值，再根据值返回索引
    best_attr = gainList.index(max(gainList))
    print("--------->")
    print("best_attr：" + str(best_attr))

    return best_attr


def treeGenerate(datas, attributes, tree_dict):
    """
    生成决策树
    :param datas: 当前数据集
    :param attributes: 当前属性集
    :param tree_dict：树字典，用来存放树结构
    :return:tree_dict：返回生成后的树结构
    """
    print("======================")
    print(datas)
    print(attributes)
    print("======================")

    # D[:,-1]取出每一列最后一个元素，即样本集D的标签，判断是否所有样本全属于同一类别C
    if allElementsEqual(datas[:,-1]):
        # 所有样本全属于同一类别C，将node标记为C类叶结点；
        # D[：-1][0]取出C类值
        node = datas[:,-1][0]
        # 返回这个结点的类别
        return node

    # 判断属性集A是否为空集 或者 D中样本在A上取值是否相同
    elif len(attributes) == 0 or allElementsEqual(datas, 2):
        # 将类别标记为D中样本数最多的类
        # np.bincount：首先找到数组最大值max，然后返回0～max的各个数字出现的次数
        # datas[:,-1]的数据类型是object，直接传入np.bincount会出现错误，因此调用.astype(np.int)转换为int类型的数据
        # np.argmax：返回数组中最大值对应的下标
        # 由于标签值刚好是0-max之间的整数值，而np.bincount每个索引其实就是原始数组的每个元素，每个索引对应的值就是相应数字出现的次数
        node = np.argmax(np.bincount(datas[:,-1].astype(np.int)))
        # 返回这个结点的内容
        return node

    else:
        # 从属性集attributes中选择最优划分属性attribute
        best_attribute = bestAttribute(datas,attributes)
        for attribute_value in attributes[best_attribute]:
            # 生成datas中在最佳划分属性best_attribute上取值为attribute_value的样本子集datasv
            datasv = datas[np.where(datas[:,best_attribute]==attribute_value)]
            if datasv.size==0:
                # 样本子集datasv没有数据
                # 将分支结点标记为叶结点，其类别标记为D样本最多的类
                tree_dict[attribute_value] = np.argmax(np.bincount(datas[:, -1].astype(np.int)))
            else:
                # 样本子集有数据，递归循环
                # 为node生成一个分支
                tree_dict[attribute_value] = {}
                # 以treeGenerate(datasv, attributes\{best_attribute}
                # np.delete(datasv, best_attribute, axis=1) 删除最佳划分属性那一列的数据，生成的是一个新数组
                # attributes.copy()先复制一个列表，避免原属性列表被影响
                attributes_copy = attributes.copy()
                # 然后.pop(best_attribute)去掉最佳划分属性这个位置的属性值，即attributes_copy = attributes\{best_attribute
                attributes_copy.pop(best_attribute)
                tree_dict[attribute_value] = treeGenerate(np.delete(datasv, best_attribute, axis=1), attributes_copy, tree_dict[attribute_value])

    # 最后返回这个字典树结构
    return tree_dict

def validation(datas, train_tree, A):
    """
    利用训练生成的决策树对验证集进行验证
    :param datas: 验证集
    :param train_tree: 训练生成的决策树
    :param A: 属性集
    :return: 准确率
    """
    for attribute in A :
        # 查找决策树每一层的属性是什么
        if list(train_tree.keys())[0] in attribute:
            # 获取这个属性在属性集里的索引，这个索引在属性集和数据集是对应的
            index = A.index(attribute)
            # datas[index]获取验证数据的这个属性的属性值
            # train_tree[datas[index]]获取这个属性值的下一层
            node = train_tree[datas[index]]
            if type(node) == dict:
                # 下一层还是个字典，说明这是个分支，应该继续查找
                predict = validation(datas, node, A)
            else:
                # 不是字典，说明是个结点，直接获取判断结果
                predict = node
    return predict

if __name__ == '__main__':
    trainDatas, trainAttributes = load_datas()
    #print(trainDatas)
    #print(trainAttributes)
    tree_root = {}
    tree = treeGenerate(trainDatas, trainAttributes, tree_root)
    print(tree)
    val_data = ['qianbai', 'yingting', 'qingcui', 'qingxi', 'pingtan', 'ruannian', 0]
    print(validation(val_data, tree, trainAttributes))