import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', help='Path to the .npy file.')

args = parser.parse_args()

def vis_npy(path):
    npy_file = np.load(path, allow_pickle=True)
    print("File",path,"succesfully loaded.")
    for caption in npy_file[1:]:
        print(caption['image_id'])

if __name__ == "__main__":
   vis_npy(args.path)
