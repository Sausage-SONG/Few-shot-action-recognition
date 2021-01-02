# Public Packages
import torch                                         #
import torchvision                                   #  Torch
from torch.utils.data import Dataset      #

from vidaug import augmentors as va                  # Video Augmentation

import cv2                                           #
import numpy as np                                   #  Image
from scipy.ndimage import rotate as rotate_img       #

import random                                        #  OS
import os                                            #

WIDTH = HEIGHT = 128
# Augmentations
prob_50 = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
prob_20 = lambda aug: va.Sometimes(0.2, aug) # Used to apply augmentor with 20% probability
aug_seq = va.Sequential([
    prob_20(va.OneOf([va.GaussianBlur(2), 
                      va.InvertColor()])),
    prob_50(va.HorizontalFlip())
])
def use_aug_seq(frames):
    aug_frames = []
    for frame in frames:
        if frame is not None:
            aug_frames.append(frame)
    
    aug_frames = aug_seq(aug_frames)
    j = 0
    for i in range(len(frames)):
        if frames[i] is not None:
            frames[i] = aug_frames[j]
            j += 1
    
    return frames

class MoCoDataset(Dataset):

    def __init__(self, data_folders, split, window_num, clip_num, frame_num, min_frame_num=25, max_vid_num=0):
        self.window_num = window_num
        self.clip_num = clip_num
        self.frame_num = frame_num
        self.video_folders = []
        
        for data_folder in data_folders:
            class_names = os.listdir(data_folder) if split == None else split
            # class_names = [''] # Finegym
            
            for class_name in class_names:
                class_path = os.path.join(data_folder, class_name)
                if not os.path.exists(class_path):
                    continue

                video_names = os.listdir(class_path)
                if len(video_names) > max_vid_num and max_vid_num != 0:
                    video_names = random.sample(video_names, max_vid_num)
                for video_name in video_names:
                    video_folder = os.path.join(class_path, video_name)

                    if len(os.listdir(video_folder)) >= min_frame_num:
                        self.video_folders.append(video_folder)

    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]

        all_frames = [os.path.join(video_folder, frame_name) for frame_name in os.listdir(video_folder)]
        all_frames.sort()

        length = len(all_frames)
        stride = round((length - self.frame_num)/(self.clip_num*self.window_num-1))
        
        selected_frames = []
        for i in range(self.clip_num*self.window_num):
            selected_frames.extend(list(range(i*stride, i*stride+self.frame_num)))
        for i in range(len(selected_frames)):
            if selected_frames[i] >= length:
                selected_frames[i] = length - 1
        
        # Process frames
        processed_frames = [None] * length
        for idx in selected_frames:
            if processed_frames[idx] is None:
                frame = all_frames[idx]
                img = cv2.imread(frame)
                img = cv2.resize(img, (WIDTH, HEIGHT))   
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed_frames[idx] = img

        frames = []
        for i, frame_idx in enumerate(selected_frames):
            j = i % self.frame_num
            if j == 0:
                frames.append([])
            
            frame = processed_frames[frame_idx].copy()
            frames[-1].append(frame)
        
        processed_frames = use_aug_seq(processed_frames)
        
        aug_frames = []
        rotate = random.randint(1,5)
        angle = random.randint(-20,20)
        for i, frame_idx in enumerate(selected_frames):
            j = i % self.frame_num
            if j == 0:
                aug_frames.append([])
            
            frame = processed_frames[frame_idx].copy()
            # Rotate
            if rotate in [1,2]:
                frame = rotate_img(frame, angle, reshape=False)
            aug_frames[-1].append(frame)
        
        final_frames = [frames, aug_frames]
        final_frames = np.array(final_frames) / 127.5 - 1              # -1 to 1 # [num_frame, h, w, channel]
        final_frames = np.transpose(final_frames, (0, 1, 5, 2, 3, 4))  # [2, window*clip, RGB, frame_num, H, W]
        final_frames = torch.Tensor(final_frames.copy())

        return final_frames

class HumanNonhumanDataset(Dataset):

    def __init__(self, h_data_folders, n_data_folders, split, window_num, clip_num, frame_num, min_frame_num=25, max_vid_num=0):
        self.window_num = window_num
        self.clip_num = clip_num
        self.frame_num = frame_num
        self.video_folders = []
        
        for data_folder in h_data_folders:
            class_names = os.listdir(data_folder) if split == None else split
            # class_names = [''] # Finegym
            
            for class_name in class_names:
                class_path = os.path.join(data_folder, class_name)
                if not os.path.exists(class_path):
                    continue

                video_names = os.listdir(class_path)
                if len(video_names) > max_vid_num and max_vid_num != 0:
                    video_names = random.sample(video_names, max_vid_num)
                for video_name in video_names:
                    h_video_folder = os.path.join(class_path, video_name)
                    n_video_folder = self.find_same_video(n_data_folders, class_name, video_name)

                    if len(os.listdir(h_video_folder)) >= min_frame_num and n_video_folder is not None and len(os.listdir(n_video_folder)) >= min_frame_num:
                        self.video_folders.append([h_video_folder, n_video_folder])
    
    def find_same_video(self, data_folders, class_name, video_name):
        for data_folder in data_folders:
            video_path = os.path.join(data_folder, class_name, video_name)
            if os.path.exists(video_path):
                return video_path
        return None

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        final_frames = []

        rotate = random.randint(1,5)
        angle = random.randint(-20,20)

        for video_folder in self.video_folders[idx]:
            all_frames = [os.path.join(video_folder, frame_name) for frame_name in os.listdir(video_folder)]
            all_frames.sort()

            length = len(all_frames)
            stride = round((length - self.frame_num)/(self.clip_num*self.window_num-1))
            
            selected_frames = []
            for i in range(self.clip_num*self.window_num):
                selected_frames.extend(list(range(i*stride, i*stride+self.frame_num)))
            for i in range(len(selected_frames)):
                if selected_frames[i] >= length:
                    selected_frames[i] = length - 1
            
            # Process frames
            processed_frames = [None] * length
            for idx in selected_frames:
                if processed_frames[idx] is None:
                    frame = all_frames[idx]
                    img = cv2.imread(frame)
                    img = cv2.resize(img, (WIDTH, HEIGHT))   
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if rotate in [1,2]:
                        img = rotate_img(img, angle, reshape=False)
                    processed_frames[idx] = img

            frames = []
            for i, frame_idx in enumerate(selected_frames):
                j = i % self.frame_num
                if j == 0:
                    frames.append([])
                
                frame = processed_frames[frame_idx].copy()
                frames[-1].append(frame)
        
            final_frames.append(frames)

        final_frames = np.array(final_frames)
        shape = final_frames.shape
        tmp_shape = np.append(shape[0]*shape[1], shape[2:])
        final_frames = final_frames.reshape(tmp_shape)
        final_frames = use_aug_seq(final_frames)
        final_frames = final_frames.reshape(shape)

        final_frames = np.array(final_frames) / 127.5 - 1              # -1 to 1 # [num_frame, h, w, channel]
        final_frames = np.transpose(final_frames, (0, 1, 5, 2, 3, 4))  # [2, window*clip, RGB, frame_num, H, W]
        final_frames = torch.Tensor(final_frames.copy())

        return final_frames