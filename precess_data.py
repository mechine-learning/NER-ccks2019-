import pandas as pd
import re
import numpy as np
from collections import Counter
"""
处理规范
原发部位：Y-B,Y-I
病灶大小：S-B,S-I
转移部位：Z-B,Z-I

other : oF

type:y s z

"""

#将单个文本进行标注
def text2pos(text,ners):
    text_list = list(text)
    text_ner_list = ['O'] * len(text_list)
    for ner in ners:
        ms = re.finditer(re.escape(ner[0]), text)
        for m in ms:
            ner_type = ner[1]
            postion = m.span()
            text_ner_list[postion[0]:postion[1]] = [ner_type + '-B'] + [ner_type + '-I'] * (len(ner[0]) - 1)

            pass
        pass
    return (text_list,text_ner_list)
    pass

#文本分句
# def text_split(text):
#     text = re.split('。|；|;',text)
#     return text
#     pass

def text_split(text):
    text = str(text)
    text = re.split('。|；|;| ,|，|；|,|\t| ', text)
    return text
    pass

def get_ners_postion(data) :
    ners = get_ners(data)
    data['ners'] = ners
    # data.loc[data.bidder == 'parakeet2004', 'ners'] = ners
    # data.loc["ners"] = ners
    result = []
    for index_,row in data.iterrows():
        text = row['原文']
        ners = row['ners']
        texts = text_split(text)
        for text in texts:
            if text == '':
                continue
                pass
            text_list,pos_list = text2pos(text,ners)
            result.append((text_list,pos_list))
            pass
        pass
    return result

    # for index_,row in data.iterrows():
    #     text = row['text']
    #     ners = row['ners']
    #     text_list = list(text)
    #     text_ner_list = ['O'] * len(text_list)
    #     for ner in ners:
    #         ms = re.finditer(re.escape(ner[0]),text)
    #         for m in ms:
    #             ner_type = ner[1]
    #             postion = m.span()
    #             text_ner_list[postion[0]:postion[1]] = [ner_type+'-B']+[ner_type+'-I']*(len(ner[0])-1)
    #             pass
    #         pass
    #     ners_result.append(text_ner_list)
    #     text_list_result.append(text_list)
    #     pass
    # data['pos'] = ners_result
    # # data.to_excel('./data/concat_pos.xlsx')
    # return text_list_result,ners_result
    pass

#一个item中所有的实体
def get_ner(s,ner):
    if s is None or pd.isna(s):
        return [(None,ner)]
        pass
    return [(i,ner) for i in s.split(',') if i!= '']
    pass

#得到itmes的所有实体
#YSZ
def get_ners(data:pd.DataFrame):
    result = []
    for _,i in data.iterrows():
        result_i =[]
        result_i.extend(get_ner(i['肿瘤原发部位'],'Y'))
        result_i.extend(get_ner(i['原发病灶大小'],'S'))
        result_i.extend(get_ner(i['转移部位'],'Z'))
        result_i = [j for j in result_i if j[0] is not None]
        #排序
        result_i = sorted(result_i,key=lambda x:len(x[0]))
        result.append(result_i)
        pass
    return result
    pass

if __name__ == '__main__':
    #读取数据
    dataone = pd.read_excel('./data/onetrain.xlsx')
    datatwo = pd.read_excel('./data/twotrain.xlsx')
    data = pd.concat((dataone,datatwo),axis=0,ignore_index=True)
    result = get_ners_postion(data)
    result = np.array(result)
    np.save('./data/train.npy',result)

    # # 保存
    # text = np.array(text)
    # pos = np.array(pos)
    # np.savez('./data/train.npy',text=text,pos=pos)

    #分析
    # ners = get_ners(data)
    # ners_Y = [j[0] for i in ners for j in i if j[1] == 'Y']
    # ners_Z = [j[0] for i in ners for j in i if j[1] == 'Z']
    # count_y = Counter(ners_Y)
    # count_z = Counter(ners_Z)
    #
    # print(ners_Y)
    # print(ners_Z)
    # #多少个
    # print(len(count_y.items()))
    # print(len(count_z.items()))
    # #频率
    # print(sorted(count_y.items(),key=lambda x:x[1],reverse=True))
    # print(sorted(count_z.items(),key=lambda x:x[1],reverse=True))

    pass