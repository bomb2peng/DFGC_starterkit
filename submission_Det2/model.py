import os
import os.path
import torch
import cv2
import numpy as np
from tqdm import tqdm
import PIL.Image as Image
import torchvision.transforms as Transforms
from torch.utils.data import dataset, dataloader
import torch.nn as nn
import json
###################### For using extra packages #################################
import sys
# Add the following two lines in order to use extra packages included in your submission
this_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(this_path)
# import your extra packages in your submission dir
from .efficientnet_pytorch.model import EfficientNet
from .albumentations import Compose, RandomBrightnessContrast, Normalize, ImageCompression, Resize, Downscale, \
    GaussianBlur, MultiplicativeNoise, \
    OneOf, RandomContrast, MedianBlur, RandomCrop, HorizontalFlip, MotionBlur
from .albumentations.pytorch import ToTensorV2


class FolderDataset(dataset.Dataset):
    def __init__(self, img_folder, face_info):
        self.img_folder = img_folder
        self.imgNames = sorted(os.listdir(img_folder))
        # REMEMBER to use sorted() to ensure correct match between imgNames and predictions
        # do NOT change the above two lines

        self.face_info = face_info
        '''
        self.transform = Transforms.Compose([
            Transforms.Resize((300, 300)),
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        '''
        self.transform = Compose([
                                HorizontalFlip(p=0.5),
                                Resize(300, 300),
                                OneOf([
                                    GaussianBlur(blur_limit=9, p=1),
                                    Downscale(scale_min=0.25, scale_max=0.5, interpolation=0, always_apply=False, p=1),
                                    ImageCompression(quality_lower=55, quality_upper=55, p=1)
                                ], p=1),
                                Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                ToTensorV2()
                                ])


    def __len__(self):
        return len(self.imgNames)

    def read_crop_face(self, img_name, img_folder, info):
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        img_name = os.path.splitext(img_name)[0]  # exclude image file extension (e.g. .png)
        # landms = info[img_name]['landms']
        box = info[img_name]['box']
        height, width = img.shape[:2]
        # enlarge the bbox by 1.3 and crop
        scale = 1.3
        # if len(box) == 2:
        #     box = box[0]
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(int(center_x - size_bb // 2), 0) # Check for out of bounds, x-y top left corner
        y1 = max(int(center_y - size_bb // 2), 0)
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)

        cropped_face = img[y1:y1 + size_bb, x1:x1 + size_bb]
        return cropped_face

    def __getitem__(self, idx):
        img_name = self.imgNames[idx]
        # Read-in images are full frames, maybe you need a face cropping.
        img = self.read_crop_face(img_name, self.img_folder, self.face_info)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        # img = self.transform(img)
        img = self.transform(image=img)
        img = img['image']
        return img


class Model():
    def __init__(self):
        # init and load your model here
        model = EfficientNet.from_name('efficientnet-b3')
        # print(model)
        model._fc = nn.Linear(1536, 1)
        thisDir = os.path.dirname(os.path.abspath(__file__))  # use this line to find this file's dir
        model.load_state_dict(torch.load(os.path.join(thisDir, 'weights.ckpt')))
        model.eval()
        model.cuda(0)
        self.model = model

        # determine your own batchsize based on your model size. The GPU memory is 16GB
        # relatively larger batchsize leads to faster execution.
        self.batchsize = 10

    def run(self, input_dir, json_file):
        with open(json_file, 'r') as load_f:
            json_info = json.load(load_f)
        dataset_eval = FolderDataset(input_dir, json_info)
        dataloader_eval = dataloader.DataLoader(dataset_eval, batch_size=self.batchsize,
                                                shuffle=False, num_workers=4)
        # USE shuffle=False in the above dataloader to ensure correct match between imgNames and predictions
        # Do set drop_last=False (default) in the above dataloader to ensure all images processed

        print('Detection model inferring ...')
        prediction = []
        with torch.no_grad():  # Do USE torch.no_grad()
            for imgs in tqdm(dataloader_eval):
                imgs = imgs.to('cuda:0')
                outputs = self.model(imgs)
                preds = torch.sigmoid(outputs)
                prediction.append(preds)

        prediction = torch.cat(prediction, dim=0)
        prediction = prediction.cpu().numpy()
        prediction = prediction.squeeze().tolist()
        assert isinstance(prediction, list)
        assert isinstance(dataset_eval.imgNames, list)
        assert len(prediction) == len(dataset_eval.imgNames)

        return dataset_eval.imgNames, prediction
