import multiprocessing


class LastOnlyQueue:

    def __init__(self, maxsize=3):
        self.maxsize = maxsize
        self.queue = multiprocessing.Queue(maxsize=maxsize)

    def put(self, item):
        while self.queue.qsize() >= self.maxsize:
            try:
                self.queue.get_nowait()
                # print("discard frame")
            except:
                break
        # while not self.queue.empty():
        #     try:
        #         self.queue.get_nowait()
        #         print("discard frame")
            # except:
            #     break
        self.queue.put(item)

    def get(self):
        return self.queue.get()

    def empty(self):
        return self.queue.empty()

    def qsize(self):
        return self.queue.qsize()
