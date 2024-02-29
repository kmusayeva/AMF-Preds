import os
import argparse


def create_directory_if_not_exists(directory):

    if not os.path.exists(directory):

        os.makedirs(directory)

        print(f"Directory '{directory}' created successfully.")

    else:

        print(f"Directory '{directory}' already exists.")




