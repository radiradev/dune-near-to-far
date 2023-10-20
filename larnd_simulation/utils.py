import os
import numpy as np
def extract_run_event_from_path(path):
    # Extract the filename from the path
    filename = os.path.basename(path)
    
    # Remove the extension
    filename_without_ext = os.path.splitext(filename)[0]

    # Split the filename using the underscore delimiter
    parts = filename_without_ext.split("_")

    # Extract the run and event numbers
    run_number = int(parts[0])
    event_number = int(parts[1])

    return run_number, event_number


def load_file(event_number, run_number):
    root = '/mnt/rradev/larndsim/converted_data'
    path = f'{root}/{run_number}_{event_number}.npz'
    sample = np.load(path)
    return sample