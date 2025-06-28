import pickle

from Model.InnerStreamFunctions import resample_signal

with open("../../S4.pkl", "rb") as file:
    data = pickle.load(file, encoding="bytes")

    #print(data[b'signal'][b'wrist'][b'EDA'])
    eda_chest = data[b'signal'][b'chest'][b'EDA']
    eda_resampled_chest = resample_signal(eda_chest, 700, 4)
    #for i in range(0, eda_resampled_chest.__len__()):
        #print(f"Chest GSR: {eda_resampled_chest[i]}   Wrist GSR: {wrist[0]} | Label: {wrist[1]}")
    print(f"Chest Resampled length: {eda_resampled_chest.__len__()}")
    print(f"Wrist length: {data[b'signal'][b'wrist'][b'EDA'].__len__()}")
    for chest, wrist in zip(eda_resampled_chest, data[b'signal'][b'wrist'][b'EDA']):
      print(f"Chest GSR: {chest} | Wrist GSR: {wrist}")
