import datetime
import re
from pydantic import BaseModel


class BiosignalData(BaseModel):
    heart_rate: int
    gsr: float
    ppg: float
    sample_rate: float

def log_to_file(obj, log_file):
    log_file = open(log_file, 'a')
    timestamp = datetime.datetime.now()
    log_file.write(f"Heart Rate: {obj['heart_rate']}; {timestamp}\n")
    log_file.write(f"PPG: {obj['ppg']}; {timestamp}\n")
    log_file.write(f"GSR: {obj['gsr']}; {timestamp}\n")
    log_file.write(f"GSR: {obj['sample_rate']}; {timestamp}\n")



def parse_file(filename, num_entries=None):
    """
    Reads the first num_entries (each consisting of 3 lines) from the file,
    processes them into BiosignalData objects, and removes these lines from the file.

    :param filename: Name of the file to be parsed.
    :param num_entries: Number of objects to read from the file (each object is 3 lines).
    :return: sensor_data_list: List of the read BiosignalData objects.
    """
    sensor_data_list = []
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Determine how many lines to process (each entry consists of 3 lines)
    if num_entries is None:
        max_lines = len(lines)
    else:
        max_lines = min(len(lines), num_entries * 3)

    # Process each complete block of 3 lines
    for i in range(0, max_lines, 3):
        if i + 2 < max_lines:
            hr_match = re.match(r"Heart Rate: (\d+); (.+)", lines[i].strip())
            psr_match = re.match(r"PPG: ([\d\.]+); (.+)", lines[i + 1].strip())
            gsr_match = re.match(r"GSR: ([\d\.]+); (.+)", lines[i + 2].strip())

            if hr_match and psr_match and gsr_match:
                heart_rate = int(hr_match.group(1))
                psr = float(psr_match.group(1))
                gsr = float(gsr_match.group(1))
                # Optionally, you can extract the timestamp if needed:
                timestamp = hr_match.group(2)
                sensor_data_list.append(BiosignalData(heart_rate, psr, gsr))
            else:
                print(f"Skipping invalid data block at lines {i}-{i + 2}")

    # "Pop" these lines by keeping only the remaining ones
    remaining_lines = lines[max_lines:]
    with open(filename, 'w') as file:
        file.writelines(remaining_lines)

    return sensor_data_list
