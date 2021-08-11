from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import torch
import numpy as np
from skimage import measure
from torch.nn import init
from torchvision import transforms
import torchvision.transforms.functional as transformsF


def generate_filelist(file_dir):
    file_list = []
    file_path = os.path.join(file_dir, 'image')
    disp_path = os.path.join(file_dir, 'disparity')
    disp_r_path = os.path.join(file_dir, 'disparity_r')
    mask_path = os.path.join(file_dir, 'mask')
    mask_r_path = os.path.join(file_dir, 'mask_r')
    for home, dirs, files in os.walk(file_path):
        for file_ in sorted(files):
            if file_.lower().endswith('png') and file_[:3]!="000" and file_[4]=="0":
                file_list.append([os.path.join(home, file_), os.path.join(home, file_[:3]+"_1.png"), 
                                    os.path.join(home, "000_0.png"), os.path.join(home, "000_1.png"),
                                    os.path.join(disp_path, home.split("/")[-1]+".png"),
                                    os.path.join(disp_r_path, home.split("/")[-1]+".png"),
                                    os.path.join(mask_path, home.split("/")[-1]+".png"),
                                    os.path.join(mask_r_path, home.split("/")[-1]+".png"),
                                    home.split("/")[-1]+'_'+file_[:3]])
    return file_list

def generate_filelist_test(file_dir, val=False):
    file_list = []
    for home, dirs, files in os.walk(file_dir):
        for file_ in sorted(files):
            if file_.lower().endswith('png') and file_[:3]!="000" and file_[4]=="0":
                if val:
                    file_list.append([os.path.join(home, file_), os.path.join(home, file_[:3]+"_1.png"),
                                    os.path.join(home, "000_0.png"), os.path.join(home, "000_1.png"),
                                    home.split("/")[-1]+'_'+file_[:3]])
                else:
                    file_list.append([os.path.join(home, file_), os.path.join(home, file_[:3]+"_1.png"),
                                    home.split("/")[-1]+'_'+file_[:3]])
    return file_list


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, resize=False):
        super(TrainSetLoader, self).__init__()
        self.resize = resize
        self.file_list = generate_filelist(dataset_dir)
    def __getitem__(self, index):
        img_left  = Image.open(self.file_list[index][0])
        img_right = Image.open(self.file_list[index][1])
        gt_left  = Image.open(self.file_list[index][2])
        gt_right = Image.open(self.file_list[index][3])
        disp = Image.open(self.file_list[index][4])
        disp_r = Image.open(self.file_list[index][5])
        mask = Image.open(self.file_list[index][6])
        mask_r = Image.open(self.file_list[index][7])
        
        if self.resize:
            h = 576
            w = 288
            img_left  = img_left.resize((h,w))
            img_right = img_right.resize((h,w))
            gt_left  = gt_left.resize((h,w))
            gt_right = gt_right.resize((h,w))
            disp = disp.resize((h,w))
            disp_r = disp_r.resize((h,w))
            mask = mask.resize((h,w))
            mask_r = mask_r.resize((h,w))
 
        hflip = False
        if random.random()<0.5:
            img_left = transformsF.hflip(img_left)
            img_right = transformsF.hflip(img_right)
            gt_left = transformsF.hflip(gt_left)
            gt_right = transformsF.hflip(gt_right)
            disp = transformsF.hflip(disp)
            disp_r = transformsF.hflip(disp_r)
            mask = transformsF.hflip(mask)
            mask_r = transformsF.hflip(mask_r)
            hflip = True
        if random.random()<0.5:
            img_left = transformsF.vflip(img_left)
            img_right = transformsF.vflip(img_right)
            gt_left = transformsF.vflip(gt_left)
            gt_right = transformsF.vflip(gt_right)
            disp = transformsF.vflip(disp)
            disp_r = transformsF.vflip(disp_r)
            mask = transformsF.vflip(mask)
            mask_r = transformsF.vflip(mask_r)

        if hflip:
            disp_tensor = -transforms.ToTensor()(np.array(disp)).float().div(255)
            disp_r_tensor = -transforms.ToTensor()(np.array(disp_r)).float().div(255)
        else:
            disp_tensor = transforms.ToTensor()(np.array(disp)).float().div(255)
            disp_r_tensor = transforms.ToTensor()(np.array(disp_r)).float().div(255)
        
        return transforms.ToTensor()(np.array(img_left)), transforms.ToTensor()(np.array(img_right)), \
                transforms.ToTensor()(np.array(gt_left)), transforms.ToTensor()(np.array(gt_right)), \
                disp_tensor, disp_r_tensor, \
                transforms.ToTensor()(np.array(mask)), transforms.ToTensor()(np.array(mask_r))
        

    def __len__(self):
        return len(self.file_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, resize=False, val=False):
        super(TestSetLoader, self).__init__()
        self.resize = resize
        self.val = val
        self.file_list = generate_filelist_test(dataset_dir, val)
    def __getitem__(self, index):
        img_left  = Image.open(self.file_list[index][0])
        img_right = Image.open(self.file_list[index][1])
        if self.val:
            gt_left  = Image.open(self.file_list[index][2])
            gt_right = Image.open(self.file_list[index][3])
        if self.resize:
            h = 576
            w = 288
            img_left  = img_left.resize((h,w))
            img_right = img_right.resize((h,w))
            if self.val:
                gt_left  = gt_left.resize((h,w))
                gt_right = gt_right.resize((h,w))
        if self.val:
            return transforms.ToTensor()(np.array(img_left)), transforms.ToTensor()(np.array(img_right)), \
            transforms.ToTensor()(np.array(gt_left)), transforms.ToTensor()(np.array(gt_right)), \
            self.file_list[index][4]
        else:
            return transforms.ToTensor()(np.array(img_left)), transforms.ToTensor()(np.array(img_right)), self.file_list[index][2]
    def __len__(self):
        return len(self.file_list)

def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)