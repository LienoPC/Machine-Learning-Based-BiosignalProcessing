import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

def parse_training_data(filename):

    df = pd.read_csv(filename)

    train = df[df.Phase == 'train']
    valid = df[df.Phase == 'valid']

    train_loss = train['Loss'].values
    valid_loss = valid['Loss'].values

    train_accuracy = train['Accuracy'].values
    valid_accuracy = valid['Accuracy'].values

    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Loss.png')
    plt.close()


    plt.figure()
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(valid_accuracy, label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Accuracy.png')
    plt.close()


parse_training_data('../../Model/Log/Saved/WESAD_MaxFreq_4/densenet121_differential/densenet121_differential.csv')