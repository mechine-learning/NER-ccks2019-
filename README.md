# NER-ccks2019-
NER(ccks2019中文电子病历迁移学习)


参加了今年ccks2019中文医疗电子病历实体抽取的任务一中的任务二但是正确率只有60%,希望看到的大佬能给出一些建议。 谢谢

实体抽取用的是BLSTM+CRF 迁移学习用的是crfsuite 词嵌入用的是Word2Vec
main_model.py里面是训练数据和抽取新的数据样本中的实体 Tradaboost1.7是迁移学习部分
调用kashgari里面实现的模型
