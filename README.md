# NER-ccks2019-
NER(ccks2019中文电子病历迁移学习)


参加了今年ccks2019中文医疗电子病历实体抽取的任务一中的任务二但是正确率只有60%,希望看到的大佬能给出一些建议。 谢谢

实体抽取用的是BLSTM+CRF 调用kashgari里面实现的模型 迁移学习用的是crfsuite 词嵌入用的是Word2Vec
main_model.py里面是训练数据和抽取新的数据样本中的实体 Tradaboost1.7是迁移学习部分

1、数据在data文件夹中</br>
2、precess_data.py是将数据预处理，将句子按标点符号切割采用BIO标注方式</br>
3、main_model.py是训练数据和输出预测结果，采用word2Vec词嵌入方式，在colab上加载bert模型会报错</br>
4、Tradaboost1.7是迁移学习部分，采用crf模型，增加预测出错的目标域句子，减少预测出错的源域句子。做的不是很好，感觉和迁移学习的基线</br>
   有差别
</br>
我的邮箱是18095480778@163.com
