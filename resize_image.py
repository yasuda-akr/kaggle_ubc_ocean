import argparse
import os
import cv2
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

Image.MAX_IMAGE_PIXELS = None
parser = argparse.ArgumentParser()
parser.add_argument('--imgsize',   type=int,   default=2048)
parser.add_argument('--input',     type=str,   default="input file path")
parser.add_argument('--output',   type=str,   default="output file path")
args = parser.parse_args()
image = Image.open( args.input )
w, h = image.size

ratio = args.imgsize / min([w, h])
w2 = int(w*ratio)
h2 = int(h*ratio)

image = image.resize( (w2, h2) )
print(f"debug: ({w}, {h}) --> ({w2}, {h2})")
if not os.path.exists(os.path.dirname(args.output)):
    os.makedirs(os.path.dirname(args.output))
image.save( args.output )
print(f"save done.")
