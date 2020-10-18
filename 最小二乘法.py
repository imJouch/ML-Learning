import numpy as np
import struct
import random
import math
# import matplotlib.pyplot as plt


# 训练集文件
train_images_idx3_ubyte_file = '/Users/jouch/Downloads/train-images-idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = '/Users/jouch/Downloads/train-labels-idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = '/Users/jouch/Downloads/t10k-images-idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = '/Users/jouch/Downloads/t10k-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))
    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


xuanze = 1

if (xuanze == 1):
    # 训练集加载
    train_images = load_train_images()
    train_labels = load_train_labels()
    # 测试集加载
    test_images = load_test_images()
    test_labels = load_test_labels()

    print('---------------------------------------------------------')
    a = []
    for i in range(60000):
        if train_labels[i] == 0.0:
            b = 1
            a.append(b)  # 训练集中数字0的个数

    pixel_position = []
    for m in range(28):  # 对mxn像素遍历
        for n in range(28):
            t = []
            q = 1
            flag = 0
            for w in range(60000):
                if train_images[w][m, n] != 0:
                    t.append(q)
                if (len(t) >= 600):  # 找到至少600张图片像素不为0的像素点
                    flag = 1
                    break
            if (flag):
                pixel_position.append((m, n))  # 找到493维像素位置,并构成数组，索引为0到492

    # A1构造
    A1 = np.zeros((60000, 494))
    for i in range(60000):  # 对60000个数据进行操作
        A1[i][0] = 1  # 第一维是1
        for j in range(1, 494):
            m = pixel_position[j - 1][0]
            n = pixel_position[j - 1][1]
            A1[i, j] = train_images[i][m, n]  # 将第i个图片493个位置的像素放到A的第i行

    # b1构造（理论值）
    b1 = []
    for i in range(60000):
        if train_labels[i] == 0.0:
            b1.append([1])
        else:
            b1.append([-1])
    b1 = np.mat(b1)

    # 求解seita
    Q, R = np.linalg.qr(A1)  # qr分解
    seita = (np.mat(R).I) * (np.mat(Q).T) * b1

    # 求解预测的b_jian（预测值）
    b_piao = []
    for i in range(60000):
        c = A1[i] * seita
        if c >= 0:
            b_piao.append([1])
        else:
            b_piao.append([-1])
    b_jian = np.mat(b_piao)

    # 进行数据统计
    d = []
    e = []
    f = []
    g = []
    for i in range(60000):
        if (b1[i] == 1) and (b_jian[i] == 1):
            d.append([1])
        if (b1[i] == 1) and (b_jian[i] == -1):
            e.append([1])
        if (b1[i] == -1) and (b_jian[i] == 1):
            f.append([1])
        if (b1[i] == -1) and (b_jian[i] == -1):
            g.append([1])
    print('训练集统计结果')
    print(len(d), ' ', len(e), ' ', len(a))
    print(len(f), ' ', len(g), ' ', 60000 - len(a))
    print(len(d) + len(f), ' ', len(e) + len(g), ' ', 60000)
    print('错误率:', (len(e) + len(f)) / 60000)

    print('------------------------------------------------------')

    a = []
    for i in range(10000):
        if test_labels[i] == 0.0:
            b = 1
            a.append(b)  # 数字0的个数
    # 矩阵A2构建（像素位置用训练集训练出来的位置）
    A2 = np.zeros((10000, 494))
    for i in range(10000):  # 对10000个数据进行操作
        A2[i][0] = 1  # 第一维是1
        for j in range(1, 494):
            m = pixel_position[j - 1][0]
            n = pixel_position[j - 1][1]
            A2[i, j] = test_images[i][m, n]  # 将493个位置的像素放到A的第i行

    # b2构造（理论值）
    b2 = []
    for i in range(10000):
        if test_labels[i] == 0.0:
            b2.append([1])
        else:
            b2.append([-1])
    b2 = np.mat(b2)

    # seita用训练集预测的

    # 求解预测的b_jian（预测值）
    b_piao = []
    for i in range(10000):
        c = A2[i] * seita
        if c >= 0:
            b_piao.append([1])
        else:
            b_piao.append([-1])
    b_jian = np.mat(b_piao)

    # 进行数据统计

    print('加入3000个新特征后-------------------------------------------------------')
    # A11构造
    RR = np.zeros((3000, 494))
    for i in range(3000):
        for j in range(494):
            RR[i, j] = random.choice([1, -1])

    cankao = (np.mat(A1)) * (np.mat(RR).T)  # 60000x494  494x3000----60000x3000
    A11 = np.zeros((60000, 3494))
    for i in range(60000):
        for j in range(3494):
            if j < 494:
                A11[i][j] = A1[i][j]  # 前一部分不动
            if (j >= 494) and (cankao[i, j - 494] > 0):
                A11[i][j] = cankao[i, j - 494]
            if (j >= 494) and (cankao[i, j - 494] <= 0):
                A11[i][j] = 0
    # b1不变（理论值不变）
    # QR分解
    Q1, R1 = np.linalg.qr(A11)
    seita1 = (np.mat(R1).I) * (np.mat(Q1).T) * b1
    # 求解预测的b_jian（预测值）
    b_piao = []
    for i in range(60000):
        c = A11[i] * seita1
        if c >= 0:
            b_piao.append([1])
        else:
            b_piao.append([-1])
    b_jian = np.mat(b_piao)

    # 略，进行数据统计（和上方一致）

    print('---------------------------------------------------------')
    # A22构造

    cankao = (np.mat(A2)) * (np.mat(RR).T)  # 10000x494  494x3000----10000x3000
    A22 = np.zeros((10000, 3494))
    for i in range(10000):
        for j in range(3494):
            if j < 494:
                A22[i][j] = A2[i][j]  # 前一部分不动
            if (j >= 494) and (cankao[i, j - 494] > 0):
                A22[i][j] = cankao[i, j - 494]
            if (j >= 494) and (cankao[i, j - 494] <= 0):
                A22[i][j] = 0
                # b2不变（对应理论值不变）
    # seita1用训练集预测的

    # 求解预测的b_jian（预测值）
    b_piao = []
    for i in range(10000):
        c = A22[i] * seita1
        if c >= 0:
            b_piao.append([1])
        else:
            b_piao.append([-1])
    b_jian = np.mat(b_piao)


# def Jacobian1(A1, beita):
#     u = A1 * beita  # u为60000x1的矩阵，即预测值
#     # 求fai_u_dao
#     uu = np.array(u)  # 格式改一下，进行e的uu次方操作
#     fai_u_dao = []
#     for i in range(60000):
#         n2 = math.exp(uu[i]) + math.exp(-uu[i])
#         n = 4 / (n2 ** 2)
#         fai_u_dao.append([n])
#     fai_u_dao = np.mat(fai_u_dao)
#
#     # 求解函数的Jocabian矩阵 60494x494
#     Df_shang = np.zeros((60000, 494))
#     A1 = np.mat(A1)
#     for i in range(60000):
#         for j in range(494):
#             Df_shang[i, j] = fai_u_dao[i] * A1[i, j]
#
#     Df_xia = 10 * np.eye(494)
#
#     Df = np.r_[Df_shang, Df_xia]  # 拼接
#     return Df
#
#
# def ff(A1, b1, beita):  # 输出f(beita)
#     u = A1 * beita  # u为60000x1的矩阵，即预测值
#     # 求fai_u
#     uu = np.array(u)  # 格式改一下，进行e的uu次方操作
#     fai_u = []
#     for i in range(60000):
#         n1 = math.exp(uu[i]) - math.exp(-uu[i])
#         n2 = math.exp(uu[i]) + math.exp(-uu[i])
#         t = n1 / n2
#         fai_u.append([t])
#
#     fai_u = np.mat(fai_u)
#     shang = fai_u - b1
#     xia = 10 * beita
#     f1 = np.r_[shang, xia]
#     return f1
#
#
# def hanshuzhi(f, t):  # 输出函数值，便于比较
#     f_fang = 0
#     for i in range(t):
#         f_fang += (f[i]) ** 2
#     return f_fang
#
#
# xuanze = 1
#
# if (xuanze == 1):
#     # 训练集加载
#     train_images = load_train_images()
#     train_labels = load_train_labels()
#     # 测试集加载
#     test_images = load_test_images()
#     test_labels = load_test_labels()
#
#     a1 = []
#     for i in range(60000):
#         if train_labels[i] == 0.0:
#             b = 1
#             a1.append(b)  # 训练集中数字0的个数
#
#     pixel_position = []
#     for m in range(28):  # 对mxn像素遍历
#         for n in range(28):
#             t = []
#             q = 1
#             flag = 0
#             for w in range(60000):
#                 if train_images[w][m, n] != 0:
#                     t.append(q)
#                 if (len(t) >= 600):  # 找到至少600张图片像素不为0的像素点
#                     flag = 1
#                     break
#             if (flag):
#                 pixel_position.append((m, n))  # 找到493维像素位置,并构成数组，索引为0到492
#
#     # A1构造
#     A1 = np.zeros((60000, 494))
#     for i in range(60000):  # 对60000个数据进行操作
#         A1[i][0] = 1  # 第一维是1
#         for j in range(1, 494):
#             m = pixel_position[j - 1][0]
#             n = pixel_position[j - 1][1]
#             A1[i, j] = train_images[i][m, n]  # 将第i张图片493个位置的像素放到A的第i行
#
#     # b1构造
#     b1 = []
#     for i in range(60000):
#         if train_labels[i] == 0.0:
#             b1.append([1])
#         else:
#             b1.append([-1])
#     b1 = np.mat(b1)
#
#     # beita初值设定、lamda初值设定
#     beita = []
#     for i in range(494):
#         m = random.choice([[0.001], [0.001]])
#         beita.append(m)
#     beita = np.mat(beita)
#
#     lamda = 10
#     # 开始循环迭代
#     for count in range(1, 26):
#         print('开始运行')
#         # 调用求解Jacobian矩阵
#         Df = Jacobian1(A1, beita)
#         Df = np.mat(Df)
#
#         # 调用计算f(beita)
#         f = ff(A1, b1, beita)
#
#         # 调用求解求函数值 （60494维）
#         f_fang = hanshuzhi(f, 60494)
#
#         # 先构造I
#         II = np.eye(494)
#         # 进行迭代
#         Df_TDf = Df.T * Df
#         kuohao = Df_TDf + lamda * II
#         zuo = kuohao.I * (Df.T)
#         beita2 = np.mat(beita) - zuo * f
#
#         # 调用计算f(beita2)
#         f2 = ff(A1, b1, beita2)
#         # 调用计算函数值
#         f2_fang = hanshuzhi(f2, 60494)
#
#         # 判断
#         if (f2_fang < f_fang):
#             beita = beita2
#             lamda = 0.8 * lamda
#         if (f2_fang >= f_fang):
#             beita = beita
#             lamda = 2 * lamda
#         if (f2_fang < 0.001):
#             break
#         print(count, '次循环结束')
#     #    print(beita)
#     print('---------------------------------------------------------')
#     # 求解预测的b_jian
#     b_piao = []
#     for i in range(60000):
#         c = A1[i] * beita
#         if c >= 0:
#             b_piao.append([1])
#         else:
#             b_piao.append([-1])
#     b_jian = np.mat(b_piao)
#
#     # 进行数据统计
#     d = []
#     e = []
#     f = []
#     g = []
#     for i in range(60000):
#         if (b1[i] == 1) and (b_jian[i] == 1):
#             d.append([1])
#         if (b1[i] == 1) and (b_jian[i] == -1):
#             e.append([1])
#         if (b1[i] == -1) and (b_jian[i] == 1):
#             f.append([1])
#         if (b1[i] == -1) and (b_jian[i] == -1):
#             g.append([1])
#     print('训练集统计结果')
#     print(len(d), ' ', len(e), ' ', 5923)
#     print(len(f), ' ', len(g), ' ', 54077)
#     print(len(d) + len(f), ' ', len(e) + len(g), ' ', 60000)
#     print('错误率:', (len(e) + len(f)) / 60000)
#     print('---------------------------------------------------------')
#     # 矩阵A2构建
#     A2 = np.zeros((10000, 494))
#     for i in range(10000):  # 对10000个数据进行操作
#         A2[i][0] = 1  # 第一维是1
#         for j in range(1, 494):
#             m = pixel_position[j - 1][0]
#             n = pixel_position[j - 1][1]
#             A2[i, j] = test_images[i][m, n]  # 将493个位置的像素放到A的第i行
#
#     # b2构造
#     b2 = []
#     for i in range(10000):
#         if test_labels[i] == 0.0:
#             b2.append([1])
#         else:
#             b2.append([-1])
#     b2 = np.mat(b2)
#
#     # beita用训练集预测的
#
#     # 求解预测的b_jian
#     b_piao = []
#     for i in range(10000):
#         c = A2[i] * beita
#         if c >= 0:
#             b_piao.append([1])
#         else:
#             b_piao.append([-1])
#     b_jian = np.mat(b_piao)
#
#     # 进行数据统计
#     d = []
#     e = []
#     f = []
#     g = []
#     for i in range(10000):
#         if (b2[i] == 1) and (b_jian[i] == 1):
#             d.append([1])
#         if (b2[i] == 1) and (b_jian[i] == -1):
#             e.append([1])
#         if (b2[i] == -1) and (b_jian[i] == 1):
#             f.append([1])
#         if (b2[i] == -1) and (b_jian[i] == -1):
#             g.append([1])
#     print('测试集统计结果')
#     print(len(d), ' ', len(e), ' ', 980)
#     print(len(f), ' ', len(g), ' ', 9020)
#     print(len(d) + len(f), ' ', len(e) + len(g), ' ', 10000)
#     print('错误率:', (len(e) + len(f)) / 10000)
#
