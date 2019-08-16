# code by chenchiwei
# -*- coding: UTF-8 -*-
import numpy as np
# from sklearn import tree
# import csv
# import model
import pandas as pd
import precess_data
import sklearn_crfsuite
import xlsxwriter
# H 测试样本分类结果
# train_S 原域训练样本 np数组
# train_A 目标域训练样本
# label_S 原域训练样本标签
# label_A 目标域训练样本标签
# test  测试样本
# N 迭代次数
# 标签对应数字
dict ={"O": 0, "Y-B": 1, "Y-I": 2, "S-B":3, "S-I":4, "Z-B":5, "Z-I":6}
CRF = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=1e-3, max_iterations=100, all_possible_transitions=True)
# def tradaboost(train_S, train_A, label_S, label_A, test, N):
def tradaboost(train_s, train_a, N):

    # train_s是一个ndarray的类型 第一列是中文 第二列是汉字对应的标签
    # 例如 ["右", "侧", "腋", "窝", "术", "后", "，", "结", "构", "紊", "乱"]  ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    # train_a test一样
    train_s = np.array(train_s)
    train_a = np.array(train_a)
    # test = np.array(test)


    # 将train_s trian_a的中文一列取出来
    train_s_sample = train_s[:, 0]
    train_a_sample = train_a[:, 0]
    # 原域训练样本
    train_s_sample = train_s_sample.reshape((train_s_sample.shape[0], 1))
    # 目标域域训练样本
    train_a_sample = train_a_sample.reshape((train_a_sample.shape[0], 1))

    #原域训练的标签
    train_s_label = train_s[:, 1]
    #目标域的标签
    train_a_label = train_a[:, 1]


    #测试的训练样本
    # test_sample = test[:, 0]
    #测试的标签
    # test_label = test[:, 1]

    # 原域标签对应的数字
    int_s_label = []
    # 目标域标签对应的数字
    int_a_label = []

    train_s_label = train_s_label.reshape((train_s_label.shape[0], 1))
    train_a_label = train_a_label.reshape((train_a_label.shape[0], 1))

    # 将train_s对应的标签转换成对应的数字
    for temp_row in train_s_label:
        temp_list = []
        for i in temp_row[0]:
            temp_list.append(dict.get(i))
        temp_list = np.array(temp_list)
        int_s_label.append(temp_list)
    int_s_label = np.array(int_s_label).reshape((len(int_s_label), 1))

    # 将train_a对应的标签转换成对应的数字
    for temp_row in train_a_label:
        temp_list = []
        for i in temp_row[0]:
            temp_list.append(dict.get(i))
        temp_list = np.array(temp_list)
        int_a_label.append(temp_list)
    int_a_label = np.array(int_a_label).reshape((len(int_a_label), 1))

    # 测试的样本
    # test_sample = test_sample.reshape((test_sample.shape[0], 1))
    #测试的样本标签
    # test_label = test_label.reshape((test_label.shape[0], 1))

    # 将train_a的中文部分和train_s的中文部分竖直拼接起来
    trains_data = np.vstack((train_a_sample, train_s_sample))

    # 将train_a的标签部分和train_s的标签部分竖直拼接起来
    trains_label = np.vstack((train_a_label, train_s_label))

    # 目标域ndarray的数据长度 train_a_sample的形状是 (好多行, 1列)
    row_A = train_a_sample.shape[0]
    row_S = train_s_sample.shape[0]

    # row_T = test.shape[0]
    # test_data 是目标域 + 原域样本 + 测试样本
    test_data = trains_data
    # test_data = np.vstack((trains_data, test_sample))

    # 初始化权重，为每一个list的权重,例如  一个list是["右", "侧", "腋", "窝", "术", "后", "，", "结", "构", "紊", "乱"]
    weights_A = np.ones([row_A, 1]) / row_A
    weights_S = np.ones([row_S, 1]) / row_S
    weights = np.concatenate((weights_A, weights_S), axis=0)

    # bata是啥我也不知道
    bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))
    # bata = 1.2
    # 存储每次迭代的标签和bata值
    bata_T = np.zeros([1, N])
    # result_label = np.ones([row_A + row_S + row_T, N])

    # predict = np.zeros([row_T])

    print('params initial finished.')

    # 将训练样本的数据转换为[[], [], []]的形式
    temp_trains_data = trains_data.tolist()
    trains_data = []
    for i in temp_trains_data:
        trains_data.append(i[0])

    # 将训练样本的标签转换为[[], [], []]的形式
    temp_trains_label = trains_label.tolist()
    trains_label = []
    for i in temp_trains_label:
        trains_label.append(i[0])

    # 将测试样本转换为[[], [], []]的样式
    temp_test_data = test_data.tolist()
    test_data = []
    for i in temp_test_data:
        test_data.append(i[0])

    # # 将测试样本转换为[[], [], []]的形式
    # temp_test_sample = test_sample.tolist()
    # final_test_sample = []
    # for i in temp_test_sample:
    #     final_test_sample.append(i[0])
    #
    # # 将测试标签转换为[[], [], []]的形式
    # temp_test_label = test_label.tolist()
    # final_test_label = []
    # for i in temp_test_label:
    #     final_test_label.append(i[0])

    P = weights
    for i in range(N):
        P = calculate_P(weights, trains_label)
        # temp返回值为 temp[0]是对test_data的预测标签     temp[1]是每个预测标签的概率
        temp = train_classify(trains_data, trains_label, test_data, P)
        temp_predict_list = temp[0]
        # 将CRF的预测值对应到相应的数字
        predict_list = []
        for k in temp_predict_list:
            list_temp = []
            for j in k:
                list_temp.append(dict.get(j))
            predict_list.append(list_temp)

        predict_list = np.array(([np.array(i) for i in predict_list]))
        predict_list = predict_list.reshape(predict_list.shape[0], 1)


        # 计算错误率  计算的是一个目标域list的整个错误率
        # 例如 ["右", "侧", "腋", "窝", "术", "后", "，", "结", "构", "紊", "乱"]，对应的标签是['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        # CRF.predict_marginal()返回的是每个标签在"Y-I", "Y-B", "Z-B", "Z-I", "S-B", "S-I"上的概率，
        #<class 'list'>: [{'Y-B': 0.7199503501945886, 'Y-I': 0.009670471846074396, 'O': 0.2603312167884775, 'Z-B': 0.007411208248609118, 'Z-I': 0.00010705136243213665, 'S-B': 0.00031981170168539726, 'S-I': 0.002209889858133883}, {'Y-B': 0.004041013750270056, 'Y-I': 0.7287940126480363, 'O': 0.2615437283014143, 'Z-B': 0.0006048597792630737, 'Z-I': 0.0032012155888523626, 'S-B': 5.1441518312954825e-06, 'S-I': 0.0018100257803337694}, {'Y-B': 1.3813306853601647e-05, 'Y-I': 0.007186718619279283, 'O': 0.992296918810369, 'Z-B': 0.00016188761397625042, 'Z-I': 0.00020635714548094102, 'S-B': 1.2940840100661227e-05, 'S-I': 0.00012136366394133917}, {'Y-B': 3.5002690960999112e-06, 'Y-I': 0.0002462057801864644, 'O': 0.9996583327479275, 'Z-B': 1.4362608083742713e-05, 'Z-I': 7.364505687942309e-05, 'S-B': 9.206803394209466e-08, 'S-I': 3.861469793816479e-06}, {'Y-B': 5.996684503215991e-07, 'Y-I': 1.2965147816580753e-07, 'O': 0.9999841204284794, 'Z-B': 1.3050060301365998e-05, 'Z-I': 2.0364684987340034e-06, 'S-B': 2.0383854376156567e-08, 'S-I': 4.3338938550822596e-08}, {'Y-B': 0.0004947266703421337, 'Y-I': 6.034242198323347e-07, 'O': 0.9421569246717325, 'Z-B': 0.057325794513640824, 'Z-I': 1.4251404245772208e-05, 'S-B': 7.652404040295923e-06, 'S-I': 4.6911779709250474e-08}, {'Y-B': 0.0004553881251014136, 'Y-I': 0.00029137927773118693, 'O': 0.8481070070202797, 'Z-B': 0.09584729525298537, 'Z-I': 0.05529321940935397, 'S-B': 4.0158126924861914e-06, 'S-I': 1.6951018565849987e-06}, {'Y-B': 0.00022750836076397343, 'Y-I': 0.00036880733387849107, 'O': 0.8602905103457719, 'Z-B': 0.0012532723292592307, 'Z-I': 0.1378486288485755, 'S-B': 5.871454775451373e-06, 'S-I': 5.401326976191956e-06}, {'Y-B': 3.1458970559545096e-05, 'Y-I': 2.922413524187516e-05, 'O': 0.9464901786068135, 'Z-B': 0.00037184763753020996, 'Z-I': 0.053063827920436256, 'S-B': 5.147066691073114e-06, 'S-I': 8.315662728035105e-06}, {'Y-B': 0.0012299066346823252, 'Y-I': 3.191384431380424e-05, 'O': 0.9111035975993016, 'Z-B': 0.038191613200109, 'Z-I': 0.04939225144137753, 'S-B': 3.878294544637488e-05, 'S-I': 1.1934334770023403e-05}, {'Y-B': 0.0006644908681522694, 'Y-I': 0.0009031675930349779, 'O': 0.4140480498110251, 'Z-B': 0.5132492125343847, 'Z-I': 0.0710959907298458, 'S-B': 1.5852698074641878e-05, 'S-I': 2.3235765483149477e-05}, {'Y-B': 7.155392963276115e-06, 'Y-I': 6.324074935111656e-05, 'O': 0.9454038933310311, 'Z-B': 6.504718581683807e-05, 'Z-I': 0.05444983842769595, 'S-B': 5.047439452070955e-06, 'S-I': 5.777473690347925e-06}, {'Y-B': 6.359895786089001e-05, 'Y-I': 6.556194878019265e-06, 'O': 0.9691999046430639, 'Z-B': 0.00039286464718958175, 'Z-I': 0.030330157275193108, 'S-B': 1.7361608368973541e-06, 'S-I': 5.182120978378844e-06}, {'Y-B': 8.112231164505198e-05, 'Y-I': 2.0818182190397964e-05, 'O': 0.9805800169258264, 'Z-B': 0.000670394117544254, 'Z-I': 0.018642739041057465, 'S-B': 1.4063989253424599e-06, 'S-I': 3.5030228117632885e-06}, {'Y-B': 6.468506283495583e-05, 'Y-I': 2.8441974943748598e-05, 'O': 0.9857829201557043, 'Z-B': 0.0002617118624311976, 'Z-I': 0.013859630549754072, 'S-B': 7.66656780859211e-07, 'S-I': 1.8437375515546128e-06}, {'Y-B': 2.430863712867711e-06, 'Y-I': 1.2776642844315683e-06, 'O': 0.9997760615110606, 'Z-B': 9.868550776806615e-06, 'Z-I': 0.0002102465488744221, 'S-B': 4.13422303832967e-08, 'S-I': 7.351906135948529e-08}, {'Y-B': 2.1978638707796518e-07, 'Y-I': 5.1510246864138675e-09, 'O': 0.9999985803626171, 'Z-B': 8.955417221939516e-07, 'Z-I': 2.9706880851651954e-07, 'S-B': 1.860027673402838e-09, 'S-I': 2.2941317416095758e-10}, {'Y-B': 3.7281309494486056e-07, 'Y-I': 1.6771831597309634e-09, 'O': 0.999998178217636, 'Z-B': 1.4418504127884939e-06, 'Z-I': 2.3845792909630636e-09, 'S-B': 3.0433129617386363e-09, 'S-I': 1.3781247309334296e-11}, {'Y-B': 2.3559619244170087e-05, 'Y-I': 1.375487841284305e-07, 'O': 0.9998799823536, 'Z-B': 9.596582033955806e-05, 'Z-I': 1.5776776865008208e-07, 'S-B': 1.9598402257649003e-07, 'S-I': 9.062412163277218e-10}, {'Y-B': 4.388636060712216e-07, 'Y-I': 7.641246657553421e-08, 'O': 0.9999975065495886, 'Z-B': 1.7677064851298734e-06, 'Z-I': 2.062799051551518e-07, 'S-B': 3.6913199714080025e-09, 'S-I': 4.966283654415124e-10}, {'Y-B': 5.254162191724883e-06, 'Y-I': 3.4390421604292724e-08, 'O': 0.9999734608637882, 'Z-B': 2.115433556889499e-05, 'Z-I': 5.247255770040467e-08, 'S-B': 4.352177831905302e-08, 'S-I': 2.536934357935887e-10}, {'Y-B': 5.070794508736342e-06, 'Y-I': 2.0975125476677626e-07, 'O': 0.9999734939830441, 'Z-B': 2.066345867140429e-05, 'Z-I': 5.183707852208254e-07, 'S-B': 4.231631583774765e-08, 'S-I': 1.3254199252146901e-09}, {'Y-B': 2.790271646802165e-07, 'Y-I': 1.0938448443417892e-08, 'O': 0.9999985759281076, 'Z-B': 1.1029383354218267e-06, 'Z-I': 2.875680269853919e-08, 'S-B': 2.3331342316585213e-09, 'S-I': 7.80067444117519e-11}, {'Y-B': 2.4077042922277962e-05, 'Y-I': 5.633502240616818e-08, 'O': 0.9999399641813992, 'Z-B': 3.574136975021069e-05, 'Z-I': 5.2043045598161e-08, 'S-B': 1.0857252985840965e-07, 'S-I': 4.5533012809920076e-10}, {'Y-B': 0.0012896982268808886, 'Y-I': 1.7776824370017057e-05, 'O': 0.9976611870287279, 'Z-B': 0.0010092024601938338, 'Z-I': 6.040738143587144e-06, 'S-B': 1.606418516586324e-05, 'S-I': 3.053651797016824e-08}]
        #每一个dict是一个标签在"Y-I", "Y-B", "Z-B", "Z-I", "S-B", "S-I"上的概率
        error_rate = 0
        predict_probability = temp[1]
        # total_A = np.sum(weights[0:row_A-1, :])
        # weights[0:row_A-1, :] = weights[0:row_A-1, :] / total_A
        # total_S = np.sum(weights[row_A: row_A+row_S-1, :])
        # weights[row_A: row_A + row_S-1, :] = weights[row_A: row_A+row_S-1, :] / total_S
        for j in range(row_A):
            a_sample = train_a_label[j, :][0]
            temp_dict_list = predict_probability[j]
            res = 0
            for k in range(len(temp_dict_list)):
                each_dict = temp_dict_list[k]
                max_probability_tuple = max(zip(each_dict.values(), each_dict.keys()))
                if max_probability_tuple[1] == a_sample[k]:
                    res = res
                else:
                    res = res + 1 - each_dict.get(a_sample[k])

            # 错误率是当前一个list每个标签的错误概率相加之后 * 该标签的权重
            error_rate = error_rate + res * weights[j, :]
        print('Error rate:', error_rate)
        # if error_rate > 0.5:
        #     error_rate = 0.5
        # if error_rate == 0:
        #     N = i
        #     break  # 防止过拟合
        #     # error_rate = 0.001

        bata_T[0, i] = np.abs(error_rate / (1 - error_rate))

        # 调整源域样本权重
        for j in range(row_S):
            power = 0
            # power = np.array(power)
            # t1 = predict_list[row_A + j]
            # t2 = int_s_label[j]
            # t4 = train_s_sample[j]
            temp_array = np.abs(predict_list[row_A + j] - int_s_label[j])
            for y in temp_array:
                power = power + np.sum(y)
            t5 = bata_T[0, i]
            if power == 0:
                power == power
            else:
                power = 0.5 / power
            t6 = np.power(bata_T[0, i], power)
            if bata_T[0, i] > 1.0:
                weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i], power)
            else:
                if power == 0:
                    weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i], power)
                else:
                    power = 0.5 / power
                    weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i], power)
            # print(str(t5) + "     " + str(t6) + "     " + str(power))
            # weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i], power)
            # print(j + "      " + str(power) + "      " + str(t5))
                # weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i], power)

            t3 = weights[row_A + j]

        # 调整目标域样本权重
        for j in range(row_A):
            power = 0
            t1 = predict_list[j]
            t2 = int_a_label[j]
            temp_array = np.abs(predict_list[j] - int_a_label[j])
            for y in temp_array:
                power = power + np.sum(y)
            if power == 0:
                power = power
            else:
                power = 1 / power
            t3 = np.power(bata, power)
            weights[j] = weights[j] * np.power(bata, power)

            t4 = weights[j]

        print("*"*100)

    # total = np.sum(weights)
    # P = weights / total
    total_A = np.sum(weights[0:row_A-1, :])
    P[0:row_A-1, :] = weights[0:row_A-1, :] / total_A
    total_S = np.sum(weights[row_A: row_A+row_S-1, :])
    P[row_A: row_A + row_S-1, :] = weights[row_A: row_A+row_S-1, :] / total_S
    # 100条目标域和2019data先出到excel文件
    row_in_BLSTMCRF = 1
    work_book = xlsxwriter.Workbook('./data/BLSTMCRF_DATA2.xlsx')
    sheet1 = work_book.add_worksheet('sheet1')
    sheet1.write(0, 0, "原文")
    sheet1.write(0, 1, "肿瘤原发部位")
    sheet1.write(0, 2, "原发病灶大小")
    sheet1.write(0, 3, "转移部位")

    # 100条目标域需要重复的样本输出成训练样本样式
    for i in range(row_A):
        sample_weight = weights[i, 0]
        repetition_times = sample_weight * (row_A + row_S)
        temp = int(np.floor(repetition_times))
        nres_list = ["Y-B", "Y-I", "S-B", "S-I", "Z-B", "Z-I"]
        how_many_nres_in_label = 0
        for j in nres_list:
            if j in trains_label[i]:
                how_many_nres_in_label = how_many_nres_in_label + 1
        if how_many_nres_in_label == 0:
            temp = temp - 5

        for k in range(temp):
            if temp < 1:
                break
            to_train_style(trains_data[i], trains_label[i], sheet1, row_in_BLSTMCRF)
            row_in_BLSTMCRF = row_in_BLSTMCRF + 1
    WTF = row_A



    # 2019data需要重复的样本输出成训练样本样式
    for i in range(row_S):
        WTF = row_A + i
        sample_weight = weights[row_A + i, 0]
        repetition_times = sample_weight * (row_A + row_S)
        temp = int(np.ceil(repetition_times))
        if temp > 5:
            temp = 3
        for j in range(temp):
            if temp < 1:
                break
            to_train_style(trains_data[row_A + i], trains_label[row_A + i], sheet1, row_in_BLSTMCRF)
            row_in_BLSTMCRF = row_in_BLSTMCRF + 1
    work_book.close()
    ddddd = WTF
    print("dsadasdsaddass")



def to_train_style(trains_data, trains_label, sheet1, row_in_BLSTMCRF):
    # 原发部位
    primary_site = []
    # 转移部位
    transfer_area = []
    # 病灶大小
    size = []

    primary_site_start_pos = []
    primary_site_end_pos = []

    transfer_area_start_pos = []
    transfer_area_end_pos = []

    size_start_pos = []
    size_end_pos = []

    # 得到原发部位的位置
    for start in range(len(trains_label)):
        if "Y-B" == trains_label[start]:
            primary_site_start_pos.append(start)
    for end in primary_site_start_pos:
        while end != len(trains_label) - 1 and trains_label[end + 1] == "Y-I":
            end = end + 1
        primary_site_end_pos.append(end)


    # 得到转移部位的位置
    for start in range(len(trains_label)):
        if "Z-B" == trains_label[start]:
            transfer_area_start_pos.append(start)
    for end in transfer_area_start_pos:
        while end != len(trains_label) - 1 and trains_label[end + 1] == "Z-I":
            end = end + 1
        transfer_area_end_pos.append(end)


    # 得到病灶大小的位置
    for start in range(len(trains_label)):
        if "S-B" == trains_label[start]:
            size_start_pos.append(start)
    for end in size_start_pos:
        while end != len(trains_label) - 1 and trains_label[end + 1] == "S-I":
            end = end + 1
        size_end_pos.append(end)

    # 得到在原文中的原发部位的文字
    for i in range(len(primary_site_start_pos)):
        temp = ""
        t1 = trains_data[primary_site_start_pos[i]: primary_site_end_pos[i] + 1]
        temp = temp.join(t1)
        primary_site.append(temp)

    # 去除重复元素
    l2 = {}.fromkeys(primary_site).keys()
    primary_site = l2

    # 得到在原文中转移部位的文字
    for i in range(len(transfer_area_start_pos)):
        temp = ""
        t1 = trains_data[transfer_area_start_pos[i]: transfer_area_end_pos[i] + 1]
        temp = temp.join(t1)
        transfer_area.append(temp)

    # 去出重复元素
    l2 = {}.fromkeys(transfer_area).keys()
    transfer_area = l2

    # 得到在原文中原发病灶大小的文字
    for i in range(len(size_start_pos)):
        temp = ""
        t1 = trains_data[size_start_pos[i]: size_end_pos[i] + 1]
        temp = temp.join(t1)
        size.append(temp)

    # 出去重复元素
    l2 = {}.fromkeys(size).keys()
    size = l2

    # 写出原文到excel
    str = ""
    str = str.join(trains_data)
    sheet1.write(row_in_BLSTMCRF, 0, str)

    # 写出原发部位到excel
    out_str = ""
    index = 0
    for text in primary_site:
        index = index + 1
        out_str = out_str + text
        if index < len(primary_site):
            out_str = out_str + ","
    sheet1.write(row_in_BLSTMCRF, 1, out_str)

    # 写出原发病灶大小到excel
    out_str = ""
    index = 0
    for text in size:
        index = index + 1
        out_str = out_str + text
        if index < len(size):
            out_str = out_str + ","
    sheet1.write(row_in_BLSTMCRF, 2, out_str)

    # 写出转移部位到excel
    out_str = ""
    index = 0
    for text in transfer_area:
        index = index + 1
        out_str = out_str + text
        if index < len(transfer_area):
            out_str = out_str + ","
    sheet1.write(row_in_BLSTMCRF, 3, out_str)


def calculate_P(weights, label):

    total = np.sum(weights)
    # weigths = weights / total
    return np.asarray(weights / total, order='C')


def train_classify(trains_data, trains_label, test_data, P):

    # clf = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random")
    CRF.fit(trains_data, trains_label)
    # BiLSTM_CRF_Model.evaluate()
    return [CRF.predict(test_data), CRF.predict_marginals(test_data)]


if __name__ == "__main__":

    dataone = pd.read_excel('./data/subtask2_training_part2.xlsx')
    datatwo = pd.read_excel('./data/data2019.xlsx')
    # datathree = pd.read_excel("./data/BLSTMCRF_DATA1.xlsx")
    # data1 = pd.concat((dataone, datatwo), axis=0, ignore_index=True)
    result1 = precess_data.get_ners_postion(dataone)
    result2 = precess_data.get_ners_postion(datatwo)
    # result3 = precess_data.get_ners_postion(datathree)

    # tradaboost(train_S, trains_A, label_S, label_A, trains_A, 50)
    # def tradaboost(train_s, train_a, test, N)
    tradaboost(result2, result1, 5)
    print("xxxxxxx")

