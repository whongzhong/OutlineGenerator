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
        req.encoding = "gb18030"
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


def get_story(url):
    html = get_msg(url)
    div = html.find("div", id="ny")
    title = html.find("div", id="content").find("h1").text
    ps = div.find_all("p")
    story = ""
    for p in ps:
        story += p.text.strip().replace("\x01", "").replace("\u3000", "")
    story = story.replace("\n", "").replace("\r", "").replace("\t", "")
    return title, story


def get_story4(url):
    html = get_msg(url)
    # print(html)
    div = html.find("div", class_="post-body")
    title = html.find("div", class_="post").find("h1").text
    ps = div.find_all("p")
    story = ""
    for idx, p in enumerate(ps[:-1]):
        if idx == 0:
            continue
        if p.find("span") or p.find("img"):
            continue
        story += p.text.strip().replace("\x01", "").replace("\u3000", "").replace("\u0005", "").replace("\u0002", "").replace("\u0006", "")
    story = story.replace("\n", "").replace("\r", "").replace("\t", "")
    return title, story


def thread_fuc3(hrefs, l):
    PREFIX = "https://www.etgushi.com"
    with open("/users10/lyzhang/Datasets/chinese_tonghua/chinese_tonghua_etgushi.jsonl", "a", encoding="utf-8") as f:
        for href in hrefs:
            try:
                url = "/".join([PREFIX, href])
                title, story = get_story(url)
                logging.info(f"get title:{title}")
                add = {"title": title.strip(), "story": story.strip()}
                f.write(json.dumps(add, ensure_ascii=False) + "\n")
            except:
                continue


def thread_fuc4(hrefs, l):
    PREFIX = "https://www.youqugushi.com"
    with open("/users10/lyzhang/Datasets/chinese_tonghua/chinese_tonghua_youqugushi.jsonl", "a", encoding="utf-8") as f:
        for href in hrefs:
            try:
                url = "/".join([PREFIX, href])
                title, story = get_story4(url)
                logging.info(f"get title:{title}")
                add = {"title": title.strip(), "story": story.strip()}
                f.write(json.dumps(add, ensure_ascii=False) + "\n")
            except:
                continue


def get_href(url):
    hrefs = []
    html = get_msg(url)
    ul = html.find("ul", class_="left-top")
    lis = ul.find_all("li")
    for li in lis:
        a = li.find("a", class_="title")
        hrefs.append(a["href"])
    return hrefs


def get_href_3(url):
    hrefs = []
    html = get_msg(url)
    ul = html.find("ul", class_="e2")
    lis = ul.find_all("li")
    for li in lis:
        a = li.find("a", class_="title")
        hrefs.append(a["href"])
    logging.info(hrefs)
    return hrefs


def main2():
    PREFIX = "https://www.etgushi.com"
    threads = []
    sub = ["jdth", "yygh", "mjgs", "etgs", "cygs", "yzgs", "sqgs", "lsgs", "thzw"]
    href = []
    for i, file in enumerate(sub):
        for j in range(150):
            url = '/'.join([PREFIX, file, f"list_{str(i+1)}_{str(j+1)}.html"])
            # logging.info(f"{url}")
            try:
                href.extend(get_href(url))
            except:
                continue
    l = 0
    r = len(href)
    logging.info(f"href len:{r}")
    num = 5
    pices = (r - l + 1) // num
    for i in range(num):
        threads.append(MyThread(thread_fuc3, (href[l: min(l+pices, r)], l)))
        l += pices
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    logging.info("END")
            
        
def main3():
    PREFIX = "https://www.youqugushi.com"
    threads = []
    sub = ["shuiqiangushi", "tonghuagushi", "shaoergushi", "youergushi", "ertongxiaogushi", "yizhigushi", "chengyugushi", "yuyangushi", "minjiangushi", "lishigushi", "youmogushi"]
    index = [1, 8, 11, 2, 3, 4, 7, 5, 6, 9, 10, 12]
    href = []
    for i, file in enumerate(sub):
        for j in range(150):
            url = '/'.join([PREFIX, file, f"list_{str(index[i])}_{str(j+1)}.html"])
            # logging.info(f"{url}")
            try:
                href.extend(get_href_3(url))
            except:
                break
    l = 0
    r = len(href)
    logging.info(f"href len:{r}")
    num = 5
    pices = (r - l + 1) // num
    for i in range(num):
        threads.append(MyThread(thread_fuc4, (href[l: min(l+pices, r)], l)))
        l += pices
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    logging.info("END")


if __name__ == "__main__":
    # a = get_msg("https://www.qigushi.com/zheligushi/0.html")
    # print(a)
    # main()
    # print(get_story("https://www.etgushi.com/etgs/44.html")[1])
    # print(get_href("https://www.etgushi.com/etgs/list_4_118.html"))
    # main2()
    # title, story = gen2("https://www.etstory.cn/tonghua/47736.html")
    # print(title)
    # story = story.replace("\u3000", "")
    # mp = {"title": title, "story": story}
    # print(json.dumps(mp, ensure_ascii=False))
    main3()
    # a, b = get_story4("https://www.youqugushi.com/p/3612.html")
    # print(a)
    # print(b)