import argparse
import os
import gc
import cv2
import math
import copy
import time
import random
import glob
from PIL import Image
from matplotlib import pyplot as plt

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torchvision

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# For Image Models
import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--imgsize',   type=int,   default=2048)
parser.add_argument('--batchsize', type=int,   default=4)
parser.add_argument('--weight',    type=str,   default=None)
parser.add_argument('--labelpkl',  type=str,   default=None)
parser.add_argument('--modelname',  type=str,   default="tf_efficientnet_b0_ns") # "convnext_small.in12k_ft_in1k_384", "tf_efficientnet_b0_ns"
parser.add_argument('--testcsv',   type=str,   default=None)
parser.add_argument('--numclasses',type=int,   default=5)

args = parser.parse_args()
print(f"args.imgsize   = {args.imgsize}")
print(f"args.batchsize = {args.batchsize}")
print(f"args.weight    = {args.weight}")
print(f"args.labelpkl  = {args.labelpkl}")
print(f"args.modelname  = {args.modelname}")
print(f"debug: args.testcsv    = {args.testcsv}")
print(f"debug: args.numclasses = {args.numclasses}")
CONFIG = {
    "seed": 42,
    "img_size": args.imgsize,
    "model_name": args.modelname,
    "num_classes": args.numclasses,
    "valid_batch_size": args.batchsize,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])
ROOT_DIR = '/kaggle/input/UBC-OCEAN'
TEST_DIR = '/kaggle/input/UBC-OCEAN/test_thumbnails'

LABEL_ENCODER_BIN = args.labelpkl
BEST_WEIGHT = args.weight
def get_test_file_path(image_id):
    return f"{TEST_DIR}/{image_id}_thumbnail.png"
if args.testcsv is None:
    df = pd.read_csv(f"{ROOT_DIR}/test.csv")
else:
    df = pd.read_csv(args.testcsv)

if not 'file_path' in df.columns:
    df['file_path'] = df['image_id'].apply(get_test_file_path)
df['label'] = 0 # dummy
df
df_sub = pd.DataFrame(df["image_id"])
encoder = joblib.load( LABEL_ENCODER_BIN )
def get_cropped_images(file_path, image_id, th_area = 1000):
    image = Image.open(file_path)
    # Aspect ratio
    as_ratio = image.size[0] / image.size[1]
    
    sxs, exs, sys, eys = [],[],[],[]
    if as_ratio >= 1.5:
        # Crop
        mask = np.max( np.array(image) > 0, axis=-1 ).astype(np.uint8)
        retval, labels = cv2.connectedComponents(mask)
        if retval >= as_ratio:
            x, y = np.meshgrid( np.arange(image.size[0]), np.arange(image.size[1]) )
            for label in range(1, retval):
                area = np.sum(labels == label)
                if area < th_area:
                    continue
                xs, ys= x[ labels == label ], y[ labels == label ]
                sx, ex = np.min(xs), np.max(xs)
                cx = (sx + ex) // 2
                crop_size = image.size[1]
                sx = max(0, cx-crop_size//2)
                ex = min(sx + crop_size - 1, image.size[0]-1)
                sx = ex - crop_size + 1
                sy, ey = 0, image.size[1]-1
                sxs.append(sx)
                exs.append(ex)
                sys.append(sy)
                eys.append(ey)
        else:
            crop_size = image.size[1]
            for i in range(int(as_ratio)):
                sxs.append( i * crop_size )
                exs.append( (i+1) * crop_size - 1 )
                sys.append( 0 )
                eys.append( crop_size - 1 )
    else:
        # Not Crop (entire image)
        sxs, exs, sys, eys = [0,],[image.size[0]-1],[0,],[image.size[1]-1]

    df_crop = pd.DataFrame()
    df_crop["image_id"] = [image_id] * len(sxs)
    df_crop["file_path"] = [file_path] * len(sxs)
    df_crop["sx"] = sxs
    df_crop["ex"] = exs
    df_crop["sy"] = sys
    df_crop["ey"] = eys
    return df_crop
dfs = []
for (file_path, image_id) in zip(df["file_path"], df["image_id"]):
    dfs.append( get_cropped_images(file_path, image_id) )

df_crop = pd.concat(dfs)
df_crop["label"] = 0 # dummy
df_crop
#df_crop = df_crop.drop_duplicates(subset=["image_id", "sx", "ex", "sy", "ey"]).reset_index(drop=True)
#df_crop
class UBCDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['label'].values
        self.transforms = transforms
        self.sxs = df["sx"].values
        self.exs = df["ex"].values
        self.sys = df["sy"].values
        self.eys = df["ey"].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        sx = self.sxs[index]
        ex = self.exs[index]
        sy = self.sys[index]
        ey = self.eys[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]
        
        img = img[ sy:ey, sx:ex, : ]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        }
data_transforms = {
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
class UBCModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super(UBCModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if "efficient" in model_name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
        else: # ConvNext
            in_features = self.model.head.in_features
            self.model.head = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.linear(pooled_features)
        return output
    
model = UBCModel(CONFIG['model_name'], CONFIG['num_classes'])
model.load_state_dict(torch.load( BEST_WEIGHT ))
model.to(CONFIG['device']);
test_dataset = UBCDataset(df_crop, transforms=data_transforms["valid"])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['valid_batch_size'], 
                          num_workers=2, shuffle=False, pin_memory=True)
preds = []
with torch.no_grad():
    bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, data in bar:        
        images = data['image'].to(CONFIG["device"], dtype=torch.float)        
        batch_size = images.size(0)
        outputs = model(images)
        outputs = model.softmax(outputs)
        preds.append( outputs.detach().cpu().numpy() )

preds = np.vstack(preds)
np.save("preds.npy", preds)
print(preds.shape)
for i in range(preds.shape[-1]):
    df_crop[f"cat{i}"] = preds[:, i]

dict_label = {}
dict_conf  = {}
for image_id, gdf in df_crop.groupby("image_id"):
    conf = gdf[ [f"cat{i}" for i in range(preds.shape[-1])] ].values.max(axis=0)
    dict_conf[image_id] = np.max(conf)
    dict_label[image_id] = np.argmax( conf )
    #dict_label[image_id] = np.argmax( gdf[ [f"cat{i}" for i in range(preds.shape[-1])] ].values.mean(axis=0) )
preds = np.array( [ dict_label[image_id] for image_id in df["image_id"].values ] )
confs = np.array( [ dict_conf[image_id] for image_id in df["image_id"].values ] )
pred_labels = encoder.inverse_transform( preds )
df_sub["label"] = pred_labels
df_sub["conf"] = confs
df_sub[["image_id", "label"]].to_csv("submission.csv", index=False)
df_sub.to_csv("log.csv", index=False)
