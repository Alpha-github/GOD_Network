#%%
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import cv2
import csv
from tqdm.notebook import tqdm
from zipfile import ZipFile

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

#%%
class PneumoniaDataset(Dataset):
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.
    
    Returns
    ------------
    images: torch.Tensor of size (B, H, W)
    bboxes: torch.Tensor of size (B, max_objects, 4)  # [ [x,y,width,height], [x,y,width,height], ...]
    '''
    def __init__(self, annotation_path, img_zip_dir, img_size=None):
        self.annotation_path = annotation_path
        self.img_zip_dir = img_zip_dir
        self.img_size = img_size
        
        self.img_data, self.bboxes= self.get_data()
        
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, idx):
        return self.img_data[idx], self.bboxes[idx], torch.tensor([1])
        
    def get_data(self):
        # Read the annotations
        with open(self.annotation_path, 'r') as file:
            reader = csv.reader(file)
            _ = next(reader)
            img_data = []
            bboxes = []

            zippedfile = ZipFile(self.img_zip_dir)

            for row in tqdm(reader):
                file_name = row[0]
                # dicom = pydicom.read_file(self.img_zip_dir+"/"+file_name+".dcm")
                dicom = pydicom.dcmread(zippedfile.open(file_name+".dcm"))
                img = apply_voi_lut(dicom.pixel_array, dicom)
                orig_h,orig_w = img.shape
                
                if self.img_size!=None:
                    img = cv2.resize(img, self.img_size)
                img_data.append(torch.from_numpy(img))
                

                if row[1]!="":
                    bbs = eval(row[1])
                    if self.img_size!=None:
                        temp = []
                        for bb in bbs:
                            x,y,w,h = bb
                            x = int(x*self.img_size[0]/orig_w)
                            y = int(y*self.img_size[1]/orig_h)
                            w = int(w*self.img_size[0]/orig_w)
                            h = int(h*self.img_size[1]/orig_h)
                            temp.append([x,y,w,h])
                        bboxes.append(torch.tensor(temp))
                    else:
                        bboxes.append(torch.tensor(bbs))
                else:
                    print("No Bounding Box")
                    break

            bboxes = pad_sequence(bboxes, batch_first=True, padding_value=-1)

            print("Created Dataset")
        return img_data, bboxes