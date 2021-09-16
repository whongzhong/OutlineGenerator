import os
import argparse
import logging
import nltk
logging.getLogger().setLevel(logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str, default="/Users/zhanglingyuan/Datasets/Gutenberg/txt")
parser.add_argument("--save_path", type=str, default="/Users/zhanglingyuan/opt/tiger/polish/data/Gutenberg.txt")


def getlen(str):
    cnt = 0
    for ch in str:
        if ch == "":
            continue
        cnt += 1
    return cnt


def prepare(path, length=50):
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


if __name__ == "__main__":
    args = parser.parse_args()
    txt_list = os.listdir(args.dir_path)
    pages = []
    for txt in txt_list:
        try:
            pages += prepare('/'.join([args.dir_path, txt]))
        except:
            pass
    with open(args.save_path, "w", encoding="utf-8") as file:
        for page in pages:
            paragraph = ""
            for word in page:
                paragraph += word + ' '
            paragraph += '\n'
            file.write(paragraph)
