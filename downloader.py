# coding: utf-8
import os
import requests


def download(filename):
    d = filename[:-4]+"_new"
    os.mkdir(d)
    with open(filename) as f:
        f.readline()
        for l in f:
            try:
                l = l.replace("\n", "").replace("\r", "")
                name, url, landmark_id = l.split(",")
                name = name[1:-1]
                url = url[1:-1]
                print(url)
                with open(os.path.join(d, landmark_id+'@'+name+'.jpg'), 'wb') as imagef:
                    imagef.write(requests.get(url).content)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    download('train.csv')
