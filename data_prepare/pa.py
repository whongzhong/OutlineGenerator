import json
from bs4 import BeautifulSoup as BS 
import requests
import logging
import time
import pandas as pd
import threading
logging.getLogger().setLevel(logging.INFO)


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


def get_msg(url):
    TIME = 0
    try:
        req = requests.get(url=url)
        req.encoding = "utf-8"
        # print(req)
        return BS(req.text, 'html.parser')
    except Exception as e:
        logging.warning('error_msg is:{} try url:{} again in 1s.'.format(e, url))
        TIME += 1
        time.sleep(1)

def gen(url):
    # url = "/".join([PREFIX, "gudaigushi", "938.html"])
    msg = get_msg(url)
    title_html = msg.find("div", class_="title")
    for item in title_html:
        if item.name == "h1":
            title = item.text
            break
    story = ""
    context = msg.find("div", class_="article_content")
    for item in context:
        if item.name == "p":
            story += item.text.strip().replace("www.qigushi.com 七故事网", "")
    return title, story


def gen2(url):
    msg = get_msg(url)
    title_html = msg.find("div", class_="story-content-header")
    title = title_html.h1.text
    story = ""
    context = msg.find("div", class_="story-content-detail")
    for item in context:
        if item.name == "p":
            story += item.text.strip()
    story = story.strip().replace("\u3000", "")
    return title, story


def thread_fuc(l, r):
    PREFIX = "https://www.qigushi.com"
    # PREFIX = "https://www.etstory.cn"
    types = ["shuiqian", "gudaigushi", "tonghuagushi", "baobao", "gelin", "ats", "1001", "yuyangushi", "chengyugushi", "zheligushi", "mingren", "aiguogushi"]
    # types = ["tonghua"]
    output = []
    with open("chinese_tonghua.jsonl", "a", encoding="utf-8") as f:
        for i in range(l, r):
            for name in types:
                try:
                    url = "/".join([PREFIX, name, f"{i}.html"])
                    title, story = gen(url)
                except Exception as e:
                    # logging.warning(f"err msg: {e}, url:{url}")
                    continue
                logging.info(f"get title:{title}")
                add = {"title": title.strip(), "story": story.strip()}
                f.write(json.dumps(add, ensure_ascii=False) + "\n")


def thread_fuc2(l ,r):
    PREFIX = "https://www.etstory.cn"
    types = ["tonghua"]
    output = []
    with open("/users10/lyzhang/Datasets/chinese_tonghua/chinese_tonghua_etstory.jsonl", "a", encoding="utf-8") as f:
        for i in range(l, r):
            for name in types:
                try:
                    url = "/".join([PREFIX, name, f"{i}.html"])
                    title, story = gen2(url)
                except Exception as e:
                    # logging.warning(f"err msg: {e}, url:{url}")
                    continue
                logging.info(f"get title:{title}")
                add = {"title": title.strip(), "story": story.strip()}
                f.write(json.dumps(add, ensure_ascii=False) + "\n")


def main():
    # PREFIX = "https://www.qigushi.com"
    # types = ["shuiqian", "gudaigushi", "tonghuagushi", "baobao", "gelin", "ats", "1001", "yuyangushi", "chengyugushi", "zheligushi", "mingren", "aiguogushi"]
    # output = []
    # with open("chinese_tonghua_new.jsonl", "a", encoding="utf-8") as f:
    #     for i in range(10000):
    #         for name in types:
    #             try:
    #                 url = "/".join([PREFIX, name, f"{i}.html"])
    #                 title, story = gen(url)
    #             except Exception as e:
    #                 logging.warning(f"err msg: {e}, url:{url}")
    #                 continue
    #             logging.info(f"get title:{title}")
    #             add = {"title": title.strip(), "story": story.strip()}
    #             f.write(json.dumps(add, ensure_ascii=False) + "\n")
    threads = []
    r = 100000
    l = 1
    num = 5
    pices = (r - l +1) // num
    for i in range(num):
        threads.append(MyThread(thread_fuc2, (l, min(l+pices, r))))
        l += pices
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    logging.info("END")


if __name__ == "__main__":
    # a = get_msg("https://www.qigushi.com/zheligushi/0.html")
    # print(a)
    main()
    # title, story = gen2("https://www.etstory.cn/tonghua/47736.html")
    # print(title)
    # story = story.replace("\u3000", "")
    # mp = {"title": title, "story": story}
    # print(json.dumps(mp, ensure_ascii=False))
