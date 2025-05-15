import multiprocessing.queues
'''
Class that models the data structure used to excange samples received from the websocket
and the data preprocessing process that prepares data to be fed to the AI model.
It is based on a shared queue that can be securely accessed by both server and data process.
'''
class DataQueueManager():

    def __init__(self, shared_queue):
        self.queue = shared_queue

    def push_single(self, obj):
        self.queue.put(obj)
        #print(f"Push queue (local proxy id): {id(self.queue)}")

    def read_batch(self, num_entries):
        read = []
        #print(f"Read queue (local proxy id): {id(self.queue)}")
        if self.queue.qsize() < num_entries:
            num_entries = self.queue.qsize()
        for i in range(0, num_entries):
            read.append(self.queue.get())

        return read