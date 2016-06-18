#!/usr/bin/python2.7
# _*_ coding: utf-8 _*_

import NavieBayes as naiveBayes
import random
import numpy as np

def testClassifyErrorRate():
    spamFileDir = './data/enron1/spam'
    hamFileDir = './data/enron1/ham'
    Words, classLables = naiveBayes.loadEMailData(spamFileDir,hamFileDir)

    # 交叉验证
    testWords = []
    testWordsType = []

    testCount = 1000
    for i in range(testCount):
        randomIndex = int(random.uniform(0, len(Words)))
        testWordsType.append(classLables[randomIndex])
        testWords.append(Words[randomIndex])
        del (Words[randomIndex])
        del (classLables[randomIndex])

    vocabularyList = naiveBayes.createVocabularyList(Words)
    print "生成语料库！"
    trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, Words)
    print "数据标记完成！"
    # 转成array向量
    trainMarkedWords = np.array(trainMarkedWords)
    print "数据转成矩阵！"
    pWordsSpam, pWordsHam, pSpam = naiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)

    errorCount = 0.0
    for i in range(testCount):
        smsType = naiveBayes.classify(vocabularyList, pWordsSpam,pWordsHam, pSpam, testWords[i])
        #print '预测类别：', smsType, '实际类别：', testWordsType[i]
        if smsType != testWordsType[i]:
            errorCount += 1

    print '错误个数：', errorCount, '错误率：', errorCount / testCount


if __name__ == '__main__':
    testClassifyErrorRate()