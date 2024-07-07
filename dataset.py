import os
from torch.utils import data
from PIL import Image

class TestDataset(data.Dataset):
    def __init__(self, path_to_txt_file, transform):
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        ta = int(self.file_list[idx].split()[1])
        sa = int(self.file_list[idx].split()[2])

        if self.transform is not None:
            img = self.transform(img)

        return img, ta, sa, image_path
    
    def __len__(self):
        return len(self.file_list)

class SplitDataset(data.Dataset):
    def __init__(self, path_to_txt_file, transform, begin, end, type = ''):
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            class_dict = {}
            for r in self.file_list:
                cls = r.split(' ')[0].split('/')[-3]
                if not (cls in class_dict):
                    class_dict[cls] = [r]
                else:
                    class_dict[cls].append(r)
            self.file_list = []
            for key in class_dict.keys():
                len_list = len(class_dict[key])
                class_dict[key] = class_dict[key][max(0, int(begin * len_list)):min(int(end * len_list), len_list)]
                self.file_list.extend(class_dict[key])

            self.file_list = [row.rstrip() for row in self.file_list]
        self.type = type
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        ta = int(self.file_list[idx].split()[1])
        sa = int(self.file_list[idx].split()[2])
        ma = int(self.file_list[idx].split()[3]) # mask attribute
        if self.transform is not None:
            img = self.transform(img)

        if self.type == 'train':
            return img, ta, sa, ma, image_path
        elif self.type == 'val':
            return img, ta, sa, image_path

    def __len__(self):
        return len(self.file_list)