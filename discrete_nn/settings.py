import os
import shutil

dataset_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../dataset_files"))

model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../model_files"))

# check if checkpoint folder exists
checkpoint_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../checkpoints"))

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)