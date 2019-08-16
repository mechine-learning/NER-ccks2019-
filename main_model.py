
#使用的是colab跑程序
#下面是挂起google 云盘
from google.colab import drive

drive.mount('/content/drive')
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

import os

os.chdir("drive/My Drive/colab")

path = get_tmpfile("word2vec.model")
# 用word2Vec做的词嵌入
model = Word2Vec(s, size=100, window=15, min_count=1, workers=4)
model.wv.save_word2vec_format("word2vec.model", binary=True)



# 这是训练部分
# author:ding xg
import numpy as np
import pandas as pd
import kashgari
import re
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model
import precess_data
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from kashgari.callbacks import EvalCallBack

if __name__ == '__main__':
    #     dataone = pd.read_excel('./data/2019data.xlsx')
    datatwo = pd.read_excel('./data/BLSTMCRF_DATA1.xlsx')
    #     datathree = pd.read_excel('./data/subtask2_training_part2.xlsx')
    # data = pd.read_excel("./data/2019data.xlsx")
    #     data = pd.concat((dataone, datatwo, datathree), axis=0, ignore_index=True)
    result = precess_data.get_ners_postion(datatwo)
    result = np.array(result)
    # train_x, test_x, train_y, test_y = train_test_split(result[:,0],  result[:,1], test_size=0.3, random_state=0)
    train_x = test_x = result[:, 0]
    train_y = test_y = result[:, 1]
    train_x = list(train_x)
    train_y = list(train_y)
    test_x = list(test_x)
    test_y = list(test_y)

    # embedding = BERTEmbedding('chinese_L-12_H-768_A-12',
    #                              task = kashgari.LABELING,
    #                              sequence_length = 110)
    word2vec_embedding = kashgari.embeddings.WordEmbedding(w2v_path="word2vec.model",
                                                           task=kashgari.LABELING,
                                                           w2v_kwargs={'binary': True, 'unicode_errors': 'ignore'},
                                                           sequence_length='auto')
    model = BiLSTM_CRF_Model(word2vec_embedding)
    tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)
    eval_callback = EvalCallBack(kash_model=model,
                                 valid_x=test_x,
                                 valid_y=test_y,
                                 step=4)

# model.build_tpu_model(strategy, train_x, train_y, test_x, test_y)
# model.compile_model()
model.fit(train_x, train_y, test_x, test_y, batch_size=20, epochs=4, callbacks=[eval_callback, tf_board_callback])
model.evaluate(test_x, test_y)
model.save('./model_8')

# 这个是写出结果到excel
import kashgari
import pandas as pd
import re

df_out = pd.DataFrame(columns=['原文', '肿瘤原发部位', '原发病灶大小', '转移部位'])
loaded_model = kashgari.utils.load_model('model_8')
"""
文件格式(head)：
text , 肿瘤原发部位, 原发病灶大小, 转移部位

表示：
肿瘤原发部位 Y
原发病灶大小 S
转移部位 Z

"""


def text_split(text):
    text = re.split('。|；|;| ,|，|；|,|\t| ', text)
    return text
    pass


def pos2ners(s, pos):
    def pos2ner_item(Type):
        result = []
        if Type + '-B' in pos:
            postions = [i for i, v in enumerate(pos) if v == Type + '-B']
            for p in postions:
                index_start = p
                index_end = index_start
                for i in range(index_start + 1, len(pos)):
                    if pos[i] != Type + '-I':
                        break
                    index_end = i
                result.append(s[index_start:index_end + 1])
                pass
            pass
        return result
        pass

    result = {}
    # 找Y 原发
    Ys = pos2ner_item('Y')
    # 找S 病灶
    Ss = pos2ner_item('S')
    # 找Z 转移
    Zs = pos2ner_item('Z')
    result['Y'] = Ys
    result['S'] = Ss
    result['Z'] = Zs
    return result


def list_to_str(L):
    ss = ""
    for i in range(len(L)):
        ss += L[i]
        if (i != len(L) - 1):
            ss += ','
    return ss


df = pd.read_excel("./data/task2 test_set_no_answer.xlsx")
for index, row in df.iterrows():
    data = row['原文']
    texts = text_split(data)
    result = loaded_model.predict(texts)
    Y = list()
    S = list()
    Z = list()
    for i in range(len(texts)):
        sentence = texts[i]
        # if len(sentence) != len(texts[i]):
        # print(sentence, texts[i])
        # print(len(sentence), len(texts[i]))
        pos = pos2ners(sentence, result[i])
        if len(pos['Y']) != 0:
            for j in pos['Y']:
                Y.append(j)
        if len(pos['S']) != 0:
            for j in pos['S']:
                S.append(j)
        if len(pos['Z']) != 0:
            for j in pos['Z']:
                Z.append(j)
        Y = list(set(Y))
        S = list(set(S))
        Z = list(set(Z))
    Y_str = list_to_str(Y)
    S_str = list_to_str(S)
    Z_str = list_to_str(Z)
    print(index, Y_str, S_str, Z_str)

    df_out = df_out.append({'原文': data, '肿瘤原发部位': Y_str, '原发病灶大小': S_str, '转移部位': Z_str}, ignore_index=True)

