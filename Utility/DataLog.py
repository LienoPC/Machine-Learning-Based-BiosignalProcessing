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

        # Create directory if doesn't exist
        os.makedirs(self.path, exist_ok=True)
        # Compuete the identifier of the new subject
        directories = os.listdir(self.path)
        new_subject_idx = len(directories) + 1

        # Create new directory
        directory_name = f"{self.path}{new_subject_idx}"
        try:
            os.mkdir(directory_name)
            self.images_dir = f"{directory_name}/images"
            os.mkdir(self.images_dir)
            # Create a csv file for raw gsr values with timestamps
            raw_path = os.path.join(directory_name, "raw.csv")
            self.raw_file = open(raw_path, "w", newline="")
            header = ["GSR", "Timestamp"]
            self.raw_writer = csv.writer(self.raw_file)
            self.raw_writer.writerow(header)

            # Create a csv file for predictions and image path with timestamps
            pred_path = os.path.join(directory_name, "predictions.csv")
            self.pred_file = open(pred_path, "w", newline="")
            header = ["Prediction", "Image", "Timestamp"]
            self.predictions_writer = csv.writer(self.pred_file)
            self.predictions_writer.writerow(header)

        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def add_raw(self, gsr, timestamp):
        self.raw_writer.writerow([gsr, timestamp])
        self.raw_file.flush()

    def add_prediction(self, prediction, image, timestamp):

        img_index = len(os.listdir(self.images_dir))
        fname = f"{self.images_dir}/{img_index}.png"
        cv2.imwrite(fname, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Add row to the csv file
        self.predictions_writer.writerow([prediction, img_index, timestamp])
        self.pred_file.flush()

    def add_prediction_ml(self, prediction, timestamp):

        save_index = len(os.listdir(self.images_dir))

        # Add row to the csv file
        self.predictions_writer.writerow([prediction, save_index, timestamp])
        self.pred_file.flush()

    def close(self):
        self.raw_file.close()
        self.pred_file.close()
