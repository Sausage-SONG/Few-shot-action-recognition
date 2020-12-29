# Public Packages
import torch                                         #
import torchvision                                   #  Torch
from torch.utils.data import DataLoader,Dataset      #
from torch.utils.data.sampler import Sampler         #

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
    # prob_20(va.OneOf([va.GaussianBlur(2), 
    #                   va.InvertColor()])),
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

# class HAADataset(Dataset):
#     def __init__(self, data_folders, mode, splits, class_num, video_num, inst_num, frame_num, clip_num, window_num):
#         self.mode = mode
#         assert mode in ["train", "test"]

#         self.class_num = class_num
#         self.video_num = video_num
#         self.inst_num = inst_num
#         self.frame_num = frame_num
#         self.clip_num = clip_num
#         self.window_num = window_num
#         self.data_folder_1 = data_folders[0]
#         self.data_folder_2 = data_folders[1]
#         self.data_folder_3 = data_folders[2]

#         all_class_names = splits[0] if self.mode == "train" else splits[1]
#         self.class_names = random.sample(all_class_names, self.class_num)
#         self.labels = dict()
#         for i, class_name in enumerate(self.class_names):
#             self.labels[class_name] = i+1

#         self.video_folders = []
#         self.video_labels = []
#         for class_name in self.class_names:
#             label = self.labels[class_name]
#             class_folders = [os.path.join(self.data_folder_1, class_name), os.path.join(self.data_folder_2, class_name), os.path.join(self.data_folder_3, class_name)]
#             video_names = os.listdir(class_folders[0])
#             random.shuffle(video_names)
#             video_names = video_names[:self.inst_num]

#             for video_name in video_names:
#                 random_stretch = random.randint(1,5)
#                 random_stretch = max(0, random_stretch-3)
#                 self.video_folders.append(os.path.join(class_folders[random_stretch], video_name))

#                 self.video_labels.append(label)
    
#     def print_dataset(self):
#         for i in range(len(self)):
#             print("[{}]\t{}\t{}".format(i, self.video_labels[i], self.video_folders[i]))
    
#     def __len__(self):
#         return len(self.video_folders)
    
#     def get_classes(self):
#         return self.class_names.copy()

#     def __getitem__(self, idx):
#         video_folder = self.video_folders[idx]
#         video_label = self.video_labels[idx]

#         all_frames = [os.path.join(video_folder, frame_name) for frame_name in os.listdir(video_folder)]
#         all_frames.sort()


#         length = len(all_frames)
#         stride = round((length - self.frame_num)/(self.clip_num*self.window_num-1))
#         expected_length = (self.clip_num*self.window_num-1)*stride + self.frame_num
        
#         # Deal with length difference
#         if expected_length <= length:
#             all_frames = all_frames[:expected_length]
#         else:
#             tmp = all_frames[-1]
#             for _ in range(expected_length - length):
#                 all_frames.append(tmp)
        
#         selected_frames = []
#         for i in range(self.clip_num*self.window_num):
#             selected_frames.extend(list(range(i*stride, i*stride+self.frame_num)))
        
#         # Process frames
#         flip = random.randint(0,1)
#         processed_frames = []
#         for frame in all_frames:
#             img = cv2.imread(frame)
#             img = cv2.resize(img, (WIDTH, HEIGHT))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.flip(img, 1) if flip else img
#             processed_frames.append(img)

#         frames = []
#         for i, frame_idx in enumerate(selected_frames):
#             j = i % self.frame_num
#             if j == 0:
#                 frames.append([])
            
#             frame = processed_frames[frame_idx].copy()
#             frames[-1].append(frame)
        
#         frames = np.array(frames) / 127.5 - 1           # -1 to 1 # [num_frame, h, w, channel]
#         frames = np.transpose(frames, (0, 4, 1, 2, 3))     # [video_clip, RGB, frame_num, H, W]
#         frames = torch.Tensor(frames.copy())

#         # noise = random.randint(0,1)
#         # if self.mode == "train" and noise:
#         #     frames = frames + 0.1 * torch.randn(self.clip_num, 3, self.frame_num, WIDTH, HEIGHT)

#         return frames, video_label

class StandardDataset(Dataset):
    def __init__(self, data_folders, mode, splits, class_num, inst_num, frame_num, clip_num, window_num):
        self.mode = mode
        assert mode in ["train", "val", "test"]

        # Attribute
        self.class_num = class_num
        self.inst_num = inst_num
        self.frame_num = frame_num
        self.clip_num = clip_num
        self.window_num = window_num
        self.data_folders = data_folders

        # Mode & Split
        if self.mode == "train":
            all_class_names = splits[0]
        elif self.mode == "val":
            all_class_names = splits[1]
        else:
            all_class_names = splits[2]
        self.class_names = random.sample(all_class_names, class_num)

        self.labels = dict()
        for i, class_name in enumerate(self.class_names):
            self.labels[class_name] = i+1

        # Find all videos
        self.video_folders = []
        self.video_labels = []
        for class_name in self.class_names:
            video_folders = []
            label = self.labels[class_name]

            for data_folder in self.data_folders:
                class_folder = os.path.join(data_folder, class_name)
                if not os.path.exists(class_folder):
                    continue
                video_names = os.listdir(class_folder) if os.path.exists(class_folder) else []

                for video_name in video_names:
                    video_path = os.path.join(class_folder, video_name)
                    if len(os.listdir(video_path)) >= self.frame_num:
                        video_folders.append(video_path)

            # Pick <self.inst_num> random videos
            video_folders = random.sample(video_folders, inst_num)
            video_labels = [label] * inst_num

            self.video_folders.extend(video_folders)
            self.video_labels.extend(video_labels)

        # self.scales = []
        # for i in range(len(self.video_folders)):
        #     self.scales.append(random.randint(2,4))

    def __len__(self):
        return len(self.video_folders)
    
    def print_dataset(self):
        string = ""
        for i in range(len(self)):
            string += "[{}] {} {} {}\n".format(i, self.video_labels[i], self.video_folders[i], self.scales[i])
        
        return string
    
    def get_labels(self):
        if self.mode == "test":
            return self.labels
        return None

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_label = self.video_labels[idx]
        # scale = self.scales[idx]

        all_frames = [os.path.join(video_folder, frame_name) for frame_name in os.listdir(video_folder)]
        all_frames.sort()
        # all_frames = all_frames[::scale]

        
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
                # # Rotate
                # if self.mode == "train" and random.randint(0,1):
                #     angle = random.randint(-25,25)
                #     img = rotate_img(img, angle, reshape=False)
                processed_frames[idx] = img
        if self.mode == 'train':
            processed_frames = use_aug_seq(processed_frames)

        frames = []
        for i, frame_idx in enumerate(selected_frames):
            j = i % self.frame_num
            if j == 0:
                frames.append([])
            
            frame = processed_frames[frame_idx].copy()
            frames[-1].append(frame)
        
        frames = np.array(frames) / 127.5 - 1              # -1 to 1 # [num_frame, h, w, channel]
        frames = np.transpose(frames, (0, 4, 1, 2, 3))     # [window*clip, RGB, frame_num, H, W]
        frames = torch.Tensor(frames.copy())

        return frames, video_label

class AVADataset(Dataset):
    def __init__(self, data_folder, mode, splits, class_num, video_num, inst_num, frame_num, clip_num, window_num):
        self.mode = mode
        assert mode in ["train", "test"]

        self.class_num = class_num
        self.video_num = video_num
        self.inst_num = inst_num
        self.frame_num = frame_num
        self.clip_num = clip_num
        self.window_num = window_num
        self.data_folder = data_folder

        all_class_names = splits[0] if self.mode == "train" else splits[1]
        while True:
            done = True
            self.class_names = random.sample(all_class_names, self.class_num)
            for class_name in self.class_names:
                class_folder = os.path.join(self.data_folder, class_name)
                if len(os.listdir(class_folder)) < self.inst_num:
                    done = False
                    break
            if done:
                break
        self.labels = dict()
        for i, class_name in enumerate(self.class_names):
            self.labels[class_name] = i+1

        self.video_folders = []
        self.video_labels = []
        for class_name in self.class_names:
            label = self.labels[class_name]
            class_folder = os.path.join(self.data_folder, class_name)
            video_names = os.listdir(class_folder)
            random.shuffle(video_names)
            video_names = video_names[:self.inst_num]

            for video_name in video_names:
                self.video_folders.append(os.path.join(class_folder, video_name))
                self.video_labels.append(label)

    def __len__(self):
        return len(self.video_folders)
    
    def print_dataset(self):
        for i in range(len(self)):
            print("[{}] {} {} {}".format(i, self.video_labels[i], self.video_folders[i], self.scales[i]))

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_label = self.video_labels[idx]

        all_frames = [os.path.join(video_folder, frame_name) for frame_name in os.listdir(video_folder)]
        all_frames.sort()

        length = len(all_frames)
        stride = round((length - self.frame_num)/(self.clip_num*self.window_num-1))
        expected_length = (self.clip_num*self.window_num-1)*stride + self.frame_num
        
        # Deal with length difference
        if expected_length <= length:
            all_frames = all_frames[:expected_length]
        else:
            tmp = all_frames[-1]
            for _ in range(expected_length - length):
                all_frames.append(tmp)
        
        selected_frames = []
        for i in range(self.clip_num*self.window_num):
            selected_frames.extend(list(range(i*stride, i*stride+self.frame_num)))
        
        # Process frames
        flip = random.randint(0,1)
        processed_frames = []
        for frame in all_frames:
            img = cv2.imread(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img, 1) if flip else img
            processed_frames.append(img)

        frames = []
        for i, frame_idx in enumerate(selected_frames):
            j = i % self.frame_num
            if j == 0:
                frames.append([])
            
            frame = processed_frames[frame_idx].copy()
            frames[-1].append(frame)
        
        frames = np.array(frames) / 127.5 - 1              # -1 to 1 # [num_frame, h, w, channel]
        frames = np.transpose(frames, (0, 4, 1, 2, 3))     # [video_clip, RGB, frame_num, H, W]
        frames = torch.Tensor(frames.copy())

        noise = random.randint(0,1)
        if self.mode == "train" and noise:
            frames = frames + 0.1 * torch.randn(self.window_num*self.clip_num, 3, self.frame_num, 128, 128)

        return frames, video_label, video_folder

class FinegymDataset(Dataset):
    def __init__(self, data_folder, info_dict, mode, splits, class_num, inst_num, frame_num, clip_num, window_num):
        self.mode = mode
        assert mode in ['train', 'val', 'test']
        
        # Attribute
        self.class_num = class_num
        self.inst_num = inst_num
        self.frame_num = frame_num
        self.clip_num = clip_num
        self.window_num = window_num
        self.data_folder = data_folder

        # Mode & Split
        if self.mode == "train":
            all_class_names = splits[0]
        elif self.mode == "val":
            all_class_names = splits[1]
        else:
            all_class_names = splits[2]
        while True:
            self.class_names = random.sample(all_class_names, class_num)
            done = True
            for class_name in self.class_names:
                if len(info_dict[class_name]) < inst_num:
                    done = False
            if done:
                break

        self.labels = dict()
        for i, class_name in enumerate(self.class_names):
            self.labels[class_name] = i+1
        
        self.video_folders = []
        self.video_labels = []
        for class_name in self.class_names:
            label = self.labels[class_name]
            video_folders = info_dict[class_name]
            video_folders = [os.path.join(data_folder, vid) for vid in video_folders]
            sample_folders = []
            for video_folder in video_folders:
                if os.path.exists(video_folder) and len(os.listdir(video_folder)) >= frame_num:
                    sample_folders.append(video_folder)
            sample_folders = random.sample(sample_folders, inst_num)

            self.video_folders.extend(sample_folders)
            self.video_labels.extend([label] * inst_num)
    
    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_label = self.video_labels[idx]

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
                # # Rotate
                # if self.mode == "train" and random.randint(0,1):
                #     angle = random.randint(-25,25)
                #     img = rotate_img(img, angle, reshape=False)
                processed_frames[idx] = img
        if self.mode == 'train':
            processed_frames = use_aug_seq(processed_frames)

        frames = []
        for i, frame_idx in enumerate(selected_frames):
            j = i % self.frame_num
            if j == 0:
                frames.append([])
            
            frame = processed_frames[frame_idx].copy()
            frames[-1].append(frame)
        
        frames = np.array(frames) / 127.5 - 1              # -1 to 1 # [num_frame, h, w, channel]
        frames = np.transpose(frames, (0, 4, 1, 2, 3))     # [window*clip, RGB, frame_num, H, W]
        frames = torch.Tensor(frames.copy())

        return frames, video_label       

# class DoubleStandardDataset(Dataset):
#     def find_same_video(self, folders, class_name, video_name):
#         for folder in folders:
#             class_path = os.path.join(folder, class_name)
#             video_path = os.path.join(class_path, video_name)
#             if os.path.exists(video_path):
#                 return video_path
#         return None

#     def __init__(self, h_data_folders, n_data_folders, mode, splits, class_num, inst_num, frame_num, clip_num, window_num):
#         self.mode = mode
#         assert mode in ["train", "val", "test"]

#         # Attribute
#         self.class_num = class_num
#         self.inst_num = inst_num
#         self.frame_num = frame_num
#         self.clip_num = clip_num
#         self.window_num = window_num

#         # Mode & Split
#         if self.mode == "train":
#             all_class_names = splits[0]
#         elif self.mode == "val":
#             all_class_names = splits[1]
#         else:
#             all_class_names = splits[2]
#         self.class_names = random.sample(all_class_names, class_num)

#         self.labels = dict()
#         for i, class_name in enumerate(self.class_names):
#             self.labels[class_name] = i+1

#         # Find all videos
#         self.video_folders = []
#         self.video_labels = []
#         for class_name in self.class_names:
#             video_folders = []
#             label = self.labels[class_name]

#             for data_folder in n_data_folders:
#                 class_folder = os.path.join(data_folder, class_name)
#                 if not os.path.exists(class_folder):
#                     continue
#                 video_names = os.listdir(class_folder) if os.path.exists(class_folder) else []

#                 for video_name in video_names:
#                     video_path = os.path.join(class_folder, video_name)
#                     if len(os.listdir(video_path)) >= self.frame_num:
#                         same_in_h = self.find_same_video(h_data_folders, class_name, video_name)
#                         if same_in_h is not None:
#                             video_folders.append([video_path, same_in_h])

#             # Pick <self.inst_num> random videos
#             video_folders = random.sample(video_folders, inst_num)
#             video_labels = [label] * inst_num

#             self.video_folders.extend(video_folders)
#             self.video_labels.extend(video_labels)

#         # self.scales = []
#         # for i in range(len(self.video_folders)):
#         #     self.scales.append(random.randint(2,4))

#     def __len__(self):
#         return len(self.video_folders)

#     def __getitem__(self, idx):
#         video_folders = self.video_folders[idx]
#         video_label = self.video_labels[idx]
#         result = []
        
#         for video_folder in video_folders:

#             all_frames = [os.path.join(video_folder, frame_name) for frame_name in os.listdir(video_folder)]
#             all_frames.sort()
#             # all_frames = all_frames[::scale]

#             length = len(all_frames)
#             stride = round((length - self.frame_num)/(self.clip_num*self.window_num-1))
            
#             selected_frames = []
#             for i in range(self.clip_num*self.window_num):
#                 selected_frames.extend(list(range(i*stride, i*stride+self.frame_num)))
#             for i in range(len(selected_frames)):
#                 if selected_frames[i] >= length:
#                     selected_frames[i] = length - 1
            
#             # Process frames
#             processed_frames = [None] * length
#             for idx in selected_frames:
#                 if processed_frames[idx] is None:
#                     frame = all_frames[idx]
#                     img = cv2.imread(frame)
#                     img = cv2.resize(img, (WIDTH, HEIGHT))   
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                     # # Rotate
#                     # if self.mode == "train" and random.randint(0,1):
#                     #     angle = random.randint(-25,25)
#                     #     img = rotate_img(img, angle, reshape=False)
#                     processed_frames[idx] = img
#             if self.mode == 'train':
#                 processed_frames = use_aug_seq(processed_frames)

#             frames = []
#             for i, frame_idx in enumerate(selected_frames):
#                 j = i % self.frame_num
#                 if j == 0:
#                     frames.append([])
                
#                 frame = processed_frames[frame_idx].copy()
#                 frames[-1].append(frame)
            
#             frames = np.array(frames) / 127.5 - 1              # -1 to 1 # [num_frame, h, w, channel]
#             frames = np.transpose(frames, (0, 4, 1, 2, 3))     # [window*clip, RGB, frame_num, H, W]

#             result.append(frames)
        
#         result = torch.Tensor(result.copy())

#         return result, video_label

class ClassBalancedSampler(Sampler):

    def __init__(self, num_per_class, class_num, inst_num, shuffle):
        self.num_per_class = num_per_class
        self.class_num = class_num
        self.inst_num = inst_num
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        batch = []
        for j in range(self.class_num):
            sublist = []
            for i in range(self.inst_num):
                sublist.append(i+j*self.inst_num)
            sublist = random.sample(sublist, self.num_per_class)
            batch.append(sublist)

        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        
        return iter(batch)

    def __len__(self):
        return 1

def get_data_loader(dataset, num_per_class, shuffle=False, num_workers=0):
    sampler = ClassBalancedSampler(num_per_class, dataset.class_num, dataset.inst_num, shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*dataset.class_num, sampler=sampler, num_workers=num_workers)
    return loader