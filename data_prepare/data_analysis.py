import os
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from utils import *
logging.getLogger().setLevel(logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/Users/zhanglingyuan/opt/tiger/polish/data/Gutenberg.txt")
parser.add_argument("--savefig_path", type=str, default="/Users/zhanglingyuan/opt/tiger/polish/log/pictures")


def count_sentence(paragragh):
    count = 0
    word_count = 0
    sentence_word_counts = []
    for word in paragragh:
        if judge_sentence(word):
            count += 1
            sentence_word_counts.append(word_count)
            word_count = 0
        else:
            word_count += 1
    return count, sentence_word_counts


def analysis_data(pages, args):
    word_counts = []
    sentence_counts = []
    sentence_word_counts = []
    for paragragh in pages:
        word_counts.append(len(paragragh))
        sentence_count, sentence_word_count = count_sentence(paragragh)
        sentence_counts.append(sentence_count)
        sentence_word_counts.extend(sentence_word_count)
    return {
        "word_counts": word_counts,
        "sentence_counts": sentence_counts,
        "sentence_word_counts": sentence_word_counts,
        "paragragh_count":len(pages)
    }


def draw_analysis(analysis, args):
    prefix = args.data_path.split("/")[-1].split(".")[0] + "_"
    word_counts_bins = np.arange(50, 450, 20)
    sentence_counts_bins = np.arange(1, 15, 1)
    sentence_word_counts_bins = np.arange(0, 200, 10)
    plt.hist(np.array(analysis["word_counts"]), bins=word_counts_bins)
    plt.xlabel("Word Counts")
    plt.ylabel("Count")
    plt.title("Word Counts Hist")
    plt.savefig("/".join([args.savefig_path, prefix + "word_counts.png"]))
    plt.cla()
    plt.hist(np.array(analysis["sentence_counts"]), bins=sentence_counts_bins)
    plt.xlabel("Sentence Counts")
    plt.ylabel("Count")
    plt.title("Sentence Counts Hist")
    plt.savefig("/".join([args.savefig_path, prefix + "sentence_counts.png"]))
    plt.cla()
    plt.hist(np.array(analysis["sentence_word_counts"]), bins=sentence_word_counts_bins)
    plt.xlabel("Sentence Word Count")
    plt.ylabel("Count")
    plt.savefig("/".join([args.savefig_path, prefix + "sentece_word_counts.png"]))
    plt.cla()
    logging.info("paragragh_count: {}k.".format(analysis["paragragh_count"] // 1000))


if __name__ == "__main__":
    args = parser.parse_args()
    pages = read_from_file(args.data_path)
    logging.info("Finished reading!")
    analysis = analysis_data(pages, args)
    logging.info("Finished get analysis!")
    draw_analysis(analysis, args)
    logging.info("End...")
