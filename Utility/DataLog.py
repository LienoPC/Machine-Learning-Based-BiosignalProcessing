import csv
import datetime
import os
import cv2

class DataLogger:
    """
    Manages data logging for the server application. For each new subject, it creates
    a subfolder and all csv files.
    """
    def __init__(self, path):
        self.path = path

        # Compuete the identifier of the new subject
        directories = os.listdir(self.path)
        new_subject_idx = len(directories) + 1

        # Create new directory
        directory_name = self.path + new_subject_idx
        try:
            os.mkdir(directory_name)

            # Create a csv file for raw gsr values with timestamps
            with open(os.path.join(directory_name, "raw.csv"), "w", newline="") as csvfile:
                header = ["GSR", "Timestamp"]
                self.raw_writer = csv.writer(csvfile)
                self.raw_writer.writerow(header)

            # Create a csv file for predictons and image path with timestamps
            with open(os.path.join(directory_name, "predictions.csv"), "w", newline="") as csvfile:
                header = ["Prediction", "Image", "Timestamp"]
                self.predictions_writer = csv.writer(csvfile)
                self.predictions_writer.writerow(header)

        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def add_raw(self, gsr, timestamp):
        self.raw_writer.writerow([gsr, timestamp])

    def add_prediction(self, prediction, image, timestamp):
        img_index = len(os.listdir(f"{self.path}/images"))
        fname = f"{self.path}/images/{img_index}.png"
        cv2.imwrite(fname, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Add row to the csv file
        self.predictions_writer.writerow([prediction, img_index, timestamp])
