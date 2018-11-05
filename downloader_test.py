
# Author: Yibo Fu
# 6/2/2018
import os
from queue import Queue
import threading
import requests


class Worker(threading.Thread):
    def __init__(self, queue, output_directory):
        threading.Thread.__init__(self)
        self.queue = queue
        self.output_directory = output_directory

    def run(self):
        while True:
            info = self.queue.get()
            self.download(info)
            self.queue.task_done()

    def download(self, info):
        info = info.replace("\n", "").replace("\r", "")
        name, url = info.split(",")
        name = name[1 : -1]
        url = url[1 : -1]

        file_path = os.path.join(self.output_directory, name + '.jpg')

        if os.path.isfile(file_path):
            return
        print(file_path)
        with open(file_path, 'wb') as f:
            f.write(requests.get(url).content)

core = 40
src = 'train_img_test.csv'
output_directory = 'cross-tests'

q = Queue()

for i in range(core):
    worker = Worker(q, output_directory)
    worker.setDaemon(True)
    worker.start()

with open(src) as file:
    file.readline() # ignore label?
    for line in file:
        q.put(file.readline())
    q.join()

