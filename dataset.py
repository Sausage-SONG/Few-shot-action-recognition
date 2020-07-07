# Public Packages
import torch                                         #
import torchvision                                   #  Torch
from torch.utils.data import DataLoader,Dataset      #
from torch.utils.data.sampler import Sampler         #

import cv2                                           #
import numpy as np                                   #  Image

import random                                        #  OS
import os                                            #

class HAADataset(Dataset):
    def __init__(self, data_folder, mode, class_num, video_num, num_inst, frame_num, seq_len):
        self.mode = mode
        assert mode in ["train", "test"]

        self.class_num = class_num
        self.video_num = video_num
        self.num_inst = num_inst
        self.frame_num = frame_num
        self.seq_len = seq_len
        self.data_folder = os.path.join(data_folder, mode)

        all_class_names = os.listdir(self.data_folder)
        self.class_names = random.sample(all_class_names, self.class_num)
        self.labels = dict()
        for i, class_name in enumerate(self.class_names):
            self.labels[class_name] = i

        self.video_folders = []
        self.video_labels = []
        for class_name in self.class_names:
            label = self.labels[class_name]
            class_folder = os.path.join(self.data_folder, class_name)
            video_names = os.listdir(class_folder)
            random.shuffle(video_names)
            video_names = video_names[:self.num_inst]
            for video_name in video_names:
                self.video_folders.append(os.path.join(class_folder, video_name))
                self.video_labels.append(label)
    
    def __str__(self):
        output = ""
        output += "Task -> mode={}; {}-way {}-shot\n".format(self.mode, self.class_num, self.video_num)
        return output
    
    def printDataset(self):
        for i in range(len(self)):
            print("[{}]\t{}\t{}".format(i, self.video_labels[i], self.video_folders[i]))
    
    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_label = self.video_labels[idx]

        all_frames = [os.path.join(video_folder, frame_name) for frame_name in os.listdir(video_folder)]
        all_frames.sort()

        i = np.random.randint(0, max(1, len(all_frames) - self.frame_num*self.seq_len))
        selected_frames = list(all_frames[i:i+self.frame_num])

        if len(selected_frames) < self.frame_num*self.seq_len:
            tmp = selected_frames[-1]
            for _ in range(self.frame_num*self.seq_len - len(selected_frames)):
                selected_frames.append(tmp)

        frames = []
        for i, frame in enumerate(selected_frames):
            j = i % self.frame_num
            if j == 0:
                frames.append([])

            img = cv2.imread(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128,128))
            frames[-1].append(img)
        
        frames = np.array(frames) / 127.5 - 1           # -1 to 1 # [num_frame, h, w, channel]
        frames = np.transpose(frames, (0, 4, 1, 2, 3))     # [video_clip, RGB, frame_num, H, W]
        frames = torch.Tensor(frames.copy())

        return frames, video_label

class ClassBalancedSampler(Sampler):

    def __init__(self, num_per_class, num_cl, num_inst):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        batch = []
        for j in range(self.num_cl):
            sublist = []
            for i in range(self.num_inst):
                sublist.append(i+j*self.num_inst)
            random.shuffle(sublist)
            sublist = sublist[:self.num_per_class]
            batch.append(sublist)

        batch = [item for sublist in batch for item in sublist]
        return iter(batch)

    def __len__(self):
        return 1

def get_HAA_data_loader(dataset, num_per_class):
    sampler = ClassBalancedSampler(num_per_class, dataset.class_num, dataset.num_inst)
    loader = DataLoader(dataset, batch_size=num_per_class*dataset.class_num, sampler=sampler, num_workers=16)
    return loader

