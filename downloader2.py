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
        name, url, landmark_id = info.split(",")
        name = name[1 : -1]
        url = url[1 : -1]

        file_path = os.path.join(self.output_directory, landmark_id + '@' + name + '.jpg')

        if os.path.isfile(file_path):
            return
        print(file_path)
        with open(file_path, 'wb') as f:
            f.write(requests.get(url).content)

core = 40
src = 'train.csv'
output_directory = 'cross-test'

q = Queue()

for i in range(core):
    worker = Worker(q, output_directory)
    worker.setDaemon(True)
    worker.start()

with open(src) as file:
    file.readline() # ignore label?
    for _ in range(100000):
        file.readline()
    for _ in range(150):
        q.put(file.readline())
    q.join()

