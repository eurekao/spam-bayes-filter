# _*_ coding: utf-8 _*_

import numpy as np
import os

def textParser(text):
    import re
    regEx = re.compile(r'[^a-zA-Z]|\d') # 匹配非字母或者数字，即去掉非字母非数字，只留下单词
    words = regEx.split(text)
    # 去除空字符串，并统一小写
    words = [word.lower() for word in words if len(word) > 0]
    return words

def loadEMailData(spamFileDir,hamFileDir):
    classCategory = []  # 类别标签，1表示是垃圾S邮件，0表示正常邮件
    Words = []
    for filename in os.listdir(spamFileDir):
        with open(spamFileDir + '/' + filename, 'r') as email:
            classCategory.append(1)
            words = textParser(email.read())
            Words.append(words)

    for filename in os.listdir(hamFileDir):
        with open(hamFileDir + '/' + filename, 'r') as email:
            classCategory.append(0)
            words = textParser(email.read())
            Words.append(words)
    return Words, classCategory

def createVocabularyList(Words):
    vocabularySet = set([])
    for words in Words:
        vocabularySet = vocabularySet | set(words)
    vocabularyList = list(vocabularySet)
    return vocabularyList

def setOfWordsToVecTor(vocabularyList, Words):
    vocabMarked = [0] * len(vocabularyList)
    for Word in Words:
        if Word in vocabularyList:
            vocabMarked[vocabularyList.index(Word)] = 1
    return vocabMarked


def setOfWordsListToVecTor(vocabularyList, WordsList):
    vocabMarkedList = []
    for i in range(len(WordsList)):
        vocabMarked = setOfWordsToVecTor(vocabularyList, WordsList[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList


def trainingNaiveBayes(trainMarkedWords, trainCategory):
    # 样本总数
    numTrainDoc = len(trainMarkedWords)
    # 特征总数
    numWords = len(trainMarkedWords[0])
    # 垃圾邮件的先验概率P(S)
    pSpam = sum(trainCategory) / (float(numTrainDoc) + numWords)
    # 对每一个特征统计其在不同分类下样本中的出现次数总和
    wordsInSpamNum = np.ones(numWords)
    wordsInHamNum = np.ones(numWords)
    spamWordsNum = 2
    HamWordsNum = 2
    for i in range(0, numTrainDoc):
        if trainCategory[i] == 1:  # 如果是垃圾邮件
            wordsInSpamNum += trainMarkedWords[i]
            spamWordsNum += 1  # 统计Spam分类下语料库中词汇出现的总次数
        else:
            wordsInHamNum += trainMarkedWords[i]
            HamWordsNum += 1
    pWordsSpam = np.log(wordsInSpamNum / spamWordsNum)
    pWordsHam = np.log(wordsInHamNum / HamWordsNum)
    return pWordsSpam, pWordsHam, pSpam

def classify(vocabularyList, pWordsSpam, pWordsHam, pSpam, testWords):
    testWordsCount = setOfWordsToVecTor(vocabularyList, testWords)
    testWordsMarkedArray = np.array(testWordsCount)
    # 计算P(Ci|W)，W为向量。P(Ci|W)只需计算P(W|Ci)P(Ci)
    p1 = sum(testWordsMarkedArray * pWordsSpam) + np.log(pSpam)
    p0 = sum(testWordsMarkedArray * pWordsHam) + np.log(1 - pSpam)
    if p1 > p0:
        return 1
    else:
        return 0