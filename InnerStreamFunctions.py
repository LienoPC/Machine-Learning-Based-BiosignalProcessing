import datetime
import re
from pydantic import BaseModel


class BiosignalData(BaseModel):
    heart_rate: int
    gsr: float
    ppg: float

def log_to_file(obj, log_file):
    log_file = open(log_file, 'a')
    timestamp = datetime.datetime.now()
    log_file.write(f"Heart Rate: {obj['heart_rate']}; {timestamp}\n")
    log_file.write(f"PPG: {obj['ppg']}; {timestamp}\n")
    log_file.write(f"GSR: {obj['gsr']}; {timestamp}\n")


def parse_file(filename, num_entries=None):
    '''

    :param filename: Name of the file to be parsed
    :param num_entries: Number of objects to read from the file
    :return: sensor_data_list: List of the read objects
    '''
    sensor_data_list = []
    with open(filename, 'r') as file:
        lines = file.readlines()

        max_lines = len(lines) if num_entries is None else min(len(lines), num_entries * 3)

        for i in range(0, max_lines, 3):  # Process every 3 lines as a group
            if i + 2 < max_lines:  # Ensure we have a complete set
                hr_match = re.match(r"Heart Rate: (\d+); (.+)", lines[i].strip())
                psr_match = re.match(r"PPG: ([\d\.]+); (.+)", lines[i + 1].strip())
                gsr_match = re.match(r"GSR: ([\d\.]+); (.+)", lines[i + 2].strip())

                if hr_match and psr_match and gsr_match:
                    heart_rate = int(hr_match.group(1))
                    psr = float(psr_match.group(1))
                    gsr = float(gsr_match.group(1))
                    timestamp = hr_match.group(2)  # Assuming all have the same timestamp

                    sensor_data_list.append(BiosignalData(heart_rate, psr, gsr))
                else:
                    print(f"Skipping invalid data block at lines {i}-{i + 2}")

    return sensor_data_list
