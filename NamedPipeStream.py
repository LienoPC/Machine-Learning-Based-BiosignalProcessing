import os
import time
import datetime
import msgpack
import re
import struct
from pydantic import BaseModel

from InnerStreamFunctions import log_to_file

# Pipe
pipe_name = r'\\.\pipe\biosignals'
log_file = "Log/Stream.txt"





def read_binary_pipe(pipe):
    """ Reads a length-prefixed MessagePack message from the named pipe """
    print("Reading message length...")
    length_data = pipe.read(4)
    if not length_data:
        return None

    message_length = struct.unpack("I", length_data)[0]  # Convert bytes to int
    print(f"Expecting {message_length} bytes...")

    data = pipe.read(message_length)
    if not data:
        return None

    return msgpack.unpackb(data)  # Deserialize MessagePack


def parse_stream_object(obj):
    '''
    PIPE PROTOCOL
    -Key 0: Heart Rate value
    -Key 1: Gsr Instant value
    -Key 2: Psr Instant value

    Update Frequency: 256Hz

    Data streamed through pipe is received as an array of values
    '''
    return DataPacket(obj[0], obj[1], obj[2])



def receiving_loop():
    with open(pipe_name, 'rb') as pipe:
        while True:
            print("Preparing to receive data\n")

            obj = read_binary_pipe(pipe)
            obj = parse_stream_object(obj)
            log_to_file(obj, log_file)
            print("Binary data received\n")


#receiving_loop()

