import logging
import nltk
import pandas as pd
import numpy as np
import random
import torch
import re


def getlen(str):
    cnt = 0
    for ch in str:
        if ch == " ":
            continue
        cnt += 1
    return cnt


def read_rawdata_new_updn(path, up=50, dn=200):
    with open(path, "r", encoding="utf-8") as file:
        txt = file.readlines()
        paragraphs = []
        for line in txt:
            word_list = line.strip().split(" ")
            if len(word_list) >= up and len(word_list) <= dn:
                paragraphs.append(line)
    return paragraphs


def read_rawdata_updn(path, up=50, dn=200):
    with open(path, "r", encoding="utf-8") as file:
        txt = file.readlines()
        paragraph = ""
        paragraphs = []
        for line in txt:
            if line == '\n':
                paragraph = nltk.word_tokenize(paragraph)
                if len(paragraph) >= up and len(paragraph) <= dn:
                    paragraphs.append(paragraph)
                paragraph = ""
            else:
                paragraph += line.strip()
    return paragraphs


def read_rawdata(path, length=50):
    with open(path, "r", encoding="utf-8") as file:
        txt = file.readlines()
        paragraph = ""
        paragraphs = []
        for line in txt:
            if line == '\n':
                paragraph = nltk.word_tokenize(paragraph)
                if len(paragraph) >= length:
                    paragraphs.append(paragraph)
                paragraph = ""
            else:
                paragraph += line.strip()
    return paragraphs


def read_data(path):
    with open(path, "r", encoding="utf-8") as file:
        txt = file.readlines()
    return txt


def read_from_file(path):
    with open(path, "r", encoding="utf-8") as file:
        pages = file.readlines()
    for i in range(len(pages)):
        pages[i] = pages[i].strip().split(" ")
    return pages


def read_from_csv(path):
    df = pd.read_csv(path)
    return df


def judge_sentence(word):
    if word.endswith(".") or word.endswith("!") or word.endswith("?"):
        return True
    if word.endswith('"'):
        if word.endswith(',"'):
            return False
        return True
    return False


def judge_in(x, minn, maxn):
    return (x >= minn and x <= maxn)


def count_word_from_sentence(sentence):
    word_list = sentence.split(" ")
    cnt = 0
    for word in word_list:
        cnt += 1
    return cnt


def debug(name, item):
    logging.debug("{}: {}".format(name, item))


def shuffle_list(list):
    idx_list = range(len(list))
    random.shuffle(idx_list)
    new_list = [list[idx] for idx in idx_list]
    return new_list


def list2str(list):
    res = ""
    for item in list:
        res += str(item)
        res += ", "
    return res[:-1]


def d2s(dt, time=False):
    if time is False:
        return dt.strftime("%Y_%m_%d")
    else:
        return dt.strftime("%Y_%m_%d_%H_%M")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed) #为当前GPU设置随机种子


def sentence_cut(text):
    text = re.sub('([\.。！？\?])([^’”])',r'\1\n\2',text)#普通断句符号且后面没有引号
    text = re.sub('(\.{6})([^’”])',r'\1\n\2',text)#英文省略号且后面没有引号
    text = re.sub('(\…{2})([^’”])',r'\1\n\2',text)#中文省略号且后面没有引号
    text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])',r'\1\n\2',text)#断句号+引号且后面没有引号
    text = text.rstrip()    # 去掉段尾的\n，然后
    return text.split("\n")


def reorder(sentence, outline):
    if len(outline) == 0:
        return outline
    order = [-1] * len(outline)
    cnt = 0
    for i in range(len(sentence)):
        for pos, word in enumerate(outline):
            if i - len(word) + 1 < 0:
                continue
            if order[pos] != -1:
                continue
            if sentence[i - len(word) + 1: i + 1] == word:
                order[pos] = cnt
                cnt += 1
    assert cnt == len(outline)
    reoutline = [outline[idx] for idx in order]
    return reoutline


def cat(sentences):
    sentence = ""
    for item in sentences:
        sentence += item
    return sentence