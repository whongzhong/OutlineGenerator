from codecs import encode
import os
import argparse
import logging
import nltk
import random
from utils import *
import pandas as pd
import numpy as np
import json
logging.getLogger().setLevel(logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str, default="/Users/zhanglingyuan/Datasets/Gutenberg/txt")
parser.add_argument("--save_path", type=str, default="/Users/zhanglingyuan/opt/tiger/polish/data/Gutenberg.txt")
parser.add_argument("--rawdata_path", type=str, default="/Users/zhanglingyuan/opt/tiger/polish/data/Gutenberg.txt")
parser.add_argument("--meta_path", type=str)
parser.add_argument("--label", type=str, default="news")
parser.add_argument("--minidata_size", type=int, default=100)
parser.add_argument("--minidata_save_path", type=str, default="/Users/zhanglingyuan/opt/tiger/polish/data")
parser.add_argument("--prepare_rawdata", action="store_true")
parser.add_argument("--prepare_minidata", action="store_true")
parser.add_argument("--cropus_type", type=str, default="Gutenberg")
parser.add_argument("--max_length", type=int, default=50)


def find_position(sentences, position):
    for i in range(max(len(sentences) - position - 1, position)):
        # logging.info(count_word_from_sentence(sentences[position + i]))
        if position + i < len(sentences) - 1 and judge_in(count_word_from_sentence(sentences[position + i]), 20, 35):
            return position + i
        if position - i > 0 and judge_in(count_word_from_sentence(sentences[position - i]), 20, 35):
            return position - i
    return None


def build_minidata(paragraphs):
    minidata = []
    idx = 1
    for paragraph in paragraphs:
        word_list = paragraph.split(" ")
        sentences = []
        sentence = ""
        for word in word_list:
            if judge_sentence(word):
                sentences.append(sentence + word)
            else:
                sentence += " " + word
        if len(sentences) >= 3:
            line = dict()
            line["id"] = idx
            line["paragraph"] = paragraph
            position = (1 + len(sentences)) // 2 -1
            position = find_position(sentences, position)
            if position is None:
                continue
            line["summarization_pos"] = position
            line["summarization_sen"] = sentences[position]
            minidata.append(line)
            idx += 1
    return minidata


def prepare_Brown_rawdata(args):
    meta_data = read_from_csv(args.meta_path)
    filename = []
    for meta in zip(meta_data["filename"], meta_data["label"]):
        if meta[1] == args.label:
            filename.append(meta[0])
    pages = []
    for file in filename:
        logging.info(file)
        txt_data = read_data("/".join([args.dir_path, file]))
        paragraph = []
        for txt in txt_data:
            if txt == "\n":
                if len(paragraph) >= args.max_length:
                    pages.append(paragraph)
                paragraph = []
            else:
                txt = txt.split(" ")
                for word in txt:
                    word_in = word.split("/")[0]
                    if word_in != "\n":
                        paragraph.append(word.split("/")[0])
    with open(args.save_path, "w", encoding="utf-8") as file:
        for page in pages:
            paragraph = ""
            for word in page:
                paragraph += word + " "
            paragraph = paragraph.strip()
            paragraph += "\n"
            file.write(paragraph)


def prepare_Gutenberg_rawdata(args):
    txt_list = os.listdir(args.dir_path)
    pages = []
    for txt in txt_list:
        try:
            pages += read_rawdata('/'.join([args.dir_path, txt]), args.max_length)
        except:
            pass
    with open(args.save_path, "w", encoding="utf-8") as file:
        for page in pages:
            paragraph = ""
            for word in page:
                paragraph += word + ' '
            paragraph += '\n'
            file.write(paragraph)


def prepare_bookcroups_rawdata(args):
    txt_list = os.listdir(args.dir_path)
    pages = []
    logging.info("txt count: {}".format(len(txt_list)))
    with open(args.save_path, "w", encoding="utf-8") as f_out:
        for txt in txt_list:
            logging.info("Start read {}.".format(txt))
            try:
                pages = read_rawdata_new_updn('/'.join([args.dir_path, txt]), args.max_length, args.max_length + 200)
                for page in pages:
                    f_out.write(page)
            except Exception as e:
                logging.warn("e:{}".format(e))


def prepare_rawdata(args):
    if args.cropus_type == "Brown":
        prepare_Brown_rawdata(args)
    elif args.cropus_type == "Gutenberg":
        prepare_Gutenberg_rawdata(args)
    elif args.cropus_type == "bookcroups":
        prepare_bookcroups_rawdata(args)


def prepare_minidata(args):
    prefix = args.rawdata_path.split("/")[-1].split(".")[0]
    paragraph = read_data(args.rawdata_path)
    random.shuffle(paragraph)
    minidata = build_minidata(paragraph[:min(len(paragraph), args.minidata_size)])
    path = "/".join([args.minidata_save_path, prefix + ".json"])
    with open(path, "w", encoding="utf-8") as file:
        for line in minidata:
            json.dump(line, file)
            file.write("\n")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.prepare_rawdata:
        logging.info("Start prepare rawdata")
        prepare_rawdata(args)
    elif args.prepare_minidata:
        logging.info("Start prepare minidata")
        prepare_minidata(args)
    logging.info("END...")
