import asyncio
import datetime


class DataQueueManager():
    '''
    Class that models the data structure used to exchange samples received from the websocket
    and the data preprocessing process that prepares data to be fed to the AI model.
    It is based on a shared queue that can be securely accessed by both server and data process.
    '''
    def __init__(self, shared_queue):
        self.queue = shared_queue
        self.window_event = asyncio.Event()
        self.window_size = None

    def push_single(self, obj):
        self.queue.append(obj)
        if self.window_size is not None:
            if len(self.queue) >= self.window_size:
                self.window_event.set()

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
        self.after_read()
        return window

    def after_read(self):
        if self.window_size is not None:
            if len(self.queue) < self.window_size:
                print("Resetting event in data queue")
                self.window_event.clear()

    async def wait_for_window(self):
        await self.window_event.wait()

    def clear(self):
        del self.queue[:]


    def set_window_samples(self, samples):
        self.window_size = samples


