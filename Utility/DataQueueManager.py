import threading


class DataQueueManager():
    '''
    Class that models the data structure used to excange samples received from the websocket
    and the data preprocessing process that prepares data to be fed to the AI model.
    It is based on a shared queue that can be securely accessed by both server and data process.
    '''
    def __init__(self, shared_queue):
        self.queue = shared_queue

    def push_single(self, obj):
        self.queue.append(obj)
        #print(f"Push queue (local proxy id): {id(self.queue)}")

    def read_batch(self, num_entries):
        num = min(num_entries, len(self.queue))
        read = list(self.queue[:num])
        del self.queue[:num]
        return read

    def read_window_overlap(self, num_entries, overlap):
        num = min(num_entries, len(self.queue))
        keep = int(num_entries*overlap)
        window = list(self.queue[:num])
        if keep <= num:
            consume = num - keep
            del self.queue[:consume]
        else:
            del self.queue[:num]

        return window

    def clear(self):
        del self.queue[:]

