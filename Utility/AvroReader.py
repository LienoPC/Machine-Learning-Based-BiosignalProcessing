import os

import neurokit2
import pandas
from fastavro import reader
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import pandas as pd

from Model.DataPreprocess.SpectrogramImages import plot_signal

csv_path = "eda_timeseries.csv"
SAMPLING_FREQ = 4.000406265258789
SAMPLE_PERIOD = timedelta(seconds=1 / SAMPLING_FREQ)

def open_single_avro(file_path):
    with open(file_path, "rb") as f:
        avro_reader = reader(f)
        records = list(avro_reader)

    df = pd.DataFrame(records)

    eda = df['rawData'].apply(lambda x: x['eda'])

    start_us = eda.iloc[0]['timestampStart']
    values = eda.iloc[0]['values']

    start_time = datetime.fromtimestamp(start_us / 1_000_000)

    timestamps = [start_time + i * SAMPLE_PERIOD for i in range(len(values))]

    return pd.DataFrame({
        "timestamp": timestamps,
        "eda_value": values
    })


def read_avro_directory(directory_path, output_csv):
    all_dfs = []
    last_timestamp = None
    continuity_issues = []

    for file_name in sorted(os.listdir(directory_path)):
        if not file_name.lower().endswith(".avro"):
            continue

        file_path = os.path.join(directory_path, file_name)
        df = open_single_avro(file_path)
        df.to_csv(f"./csv/{file_name}.csv", index=False)
        if last_timestamp is not None:
            expected_start = last_timestamp + SAMPLE_PERIOD
            actual_start = df['timestamp'].iloc[0]
            if abs((actual_start - expected_start).total_seconds()) > 1e-6:
                continuity_issues.append({
                    "prev_file_end": last_timestamp,
                    "current_file_start": actual_start,
                    "file_name": file_name,
                    "gap_seconds": (actual_start - expected_start).total_seconds()
                })

        last_timestamp = df['timestamp'].iloc[-1]
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True).sort_values("timestamp")

    final_df.to_csv(output_csv, index=False)
    print(f"Saved {len(final_df)} rows to {output_csv}")

    if continuity_issues:
        print("\nTimestamp continuity issues found between files:")
        for issue in continuity_issues:
            print(f"- File '{issue['file_name']}': "
                  f"expected start {issue['prev_file_end']} → actual start {issue['current_file_start']} "
                  f"(gap: {issue['gap_seconds']}s)")
    else:
        print("\nAll file timestamps are continuous.")


def read_csv_embrace_eda(file_list, time_start, time_end):
    dfs = []
    for file in file_list:
        df = pd.read_csv(file, parse_dates=['timestamp'])
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)

    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])

    mask = (full_df['timestamp'] >= pd.to_datetime(time_start)) & \
           (full_df['timestamp'] <= pd.to_datetime(time_end))
    filtered_df = full_df.loc[mask]

    return filtered_df['eda_value'].to_numpy()

def read_and_plot(file_list, time_start, time_end, path, sampling_rate, filter=True):
    eda = read_csv_embrace_eda(file_list, time_start, time_end)
    if filter:
        eda = neurokit2.eda.eda_clean(eda, sampling_rate=sampling_rate)
    plot_signal(eda, os.path.join(path, "RawSignal"))
    return eda


#read_avro_directory("C:\\Users\\Prova\\Downloads\\2025-07-31\\0-3YKC61P1NV\\raw_data\\v6\\Subject1", csv_path)

