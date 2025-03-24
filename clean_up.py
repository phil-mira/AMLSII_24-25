import os
import re

# Path to the directory containing the files
directory = "./data/train_images"

# Regular expression to match files ending with synth_{num}
pattern = re.compile(r"_synth_\d.jpg+$")

# Iterate through the files in the directory
for filename in os.listdir(directory):
    if pattern.search(filename):
        file_path = os.path.join(directory, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")