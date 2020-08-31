# Public Packages
import torch                                         #
import torchvision                                   #  Torch
from torch.utils.data import DataLoader,Dataset      #
from torch.utils.data.sampler import Sampler         #

import cv2                                           #
import numpy as np                                   #  Image

import random                                        #  OS
import os                                            #

WIDTH = HEIGHT = 128

class HAADataset(Dataset):
    def __init__(self, data_folders, classes, mode, class_num, video_num, inst_num, frame_num, clip_num, window_num):
        self.mode = mode
        assert mode in ["train", "test"]

        self.class_num = class_num
        self.video_num = video_num
        self.inst_num = inst_num
        self.frame_num = frame_num
        self.clip_num = clip_num
        self.window_num = window_num
        self.data_folder_1 = os.path.join(data_folders[0], mode)
        self.data_folder_2 = os.path.join(data_folders[1], mode)
        self.data_folder_3 = os.path.join(data_folders[2], mode)

        all_class_names = os.listdir(self.data_folder_1)
        self.class_names = classes if classes is not None else random.sample(all_class_names, self.class_num)
        self.labels = dict()
        for i, class_name in enumerate(self.class_names):
            self.labels[class_name] = i+1

        self.video_folders = []
        self.video_labels = []
        for class_name in self.class_names:
            label = self.labels[class_name]
            class_folders = [os.path.join(self.data_folder_1, class_name), os.path.join(self.data_folder_2, class_name), os.path.join(self.data_folder_3, class_name)]
            video_names = os.listdir(class_folders[0])
            random.shuffle(video_names)
            video_names = video_names[:self.inst_num]

            for video_name in video_names:
                random_stretch = random.randint(1,5)
                random_stretch = max(0, random_stretch-3)
                self.video_folders.append(os.path.join(class_folders[random_stretch], video_name))

                self.video_labels.append(label)
    
    def __str__(self):
        output = ""
        output += "Task -> mode={}; {}-way {}-shot\n".format(self.mode, self.class_num, self.video_num)
        return output
    
    def print_dataset(self):
        for i in range(len(self)):
            print("[{}]\t{}\t{}".format(i, self.video_labels[i], self.video_folders[i]))
    
    def __len__(self):
        return len(self.video_folders)
    
    def get_classes(self):
        return self.class_names.copy()

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
        # trans = random.randint(0,1)
        # T = np.float32([[1,0,random.randint(-25,25)],[0,1,random.randint(-25,25)]])
        processed_frames = []
        for frame in all_frames:
            img = cv2.imread(frame)
            img = cv2.resize(img, (WIDTH, HEIGHT))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img, 1) if flip else img
            # img = cv2.warpAffine(img, T, (WIDTH, HEIGHT)) if trans else img
            processed_frames.append(img)

        frames = []
        for i, frame_idx in enumerate(selected_frames):
            j = i % self.frame_num
            if j == 0:
                frames.append([])
            
            frame = processed_frames[frame_idx].copy()
            frames[-1].append(frame)
        
        frames = np.array(frames) / 127.5 - 1           # -1 to 1 # [num_frame, h, w, channel]
        frames = np.transpose(frames, (0, 4, 1, 2, 3))     # [video_clip, RGB, frame_num, H, W]
        frames = torch.Tensor(frames.copy())

        noise = random.randint(0,1)
        if self.mode == "train" and noise:
            frames = frames + 0.1 * torch.randn(self.clip_num, 3, self.frame_num, WIDTH, HEIGHT)

        return frames, video_label

class KineticsDataset(Dataset):
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
        self.class_names = random.sample(all_class_names, self.class_num)
        self.labels = dict()
        for i, class_name in enumerate(self.class_names):
            self.labels[class_name] = i+1

        self.video_folders = []
        self.video_labels = []
        self.scales = []
        for class_name in self.class_names:
            label = self.labels[class_name]
            class_folder = os.path.join(self.data_folder, class_name)
            video_names = os.listdir(class_folder)
            random.shuffle(video_names)
            video_names = video_names[:self.inst_num]

            for video_name in video_names:
                random_stretch = random.randint(1,3)
                self.video_folders.append(os.path.join(class_folder, video_name))
                self.video_labels.append(label)
                self.scales.append(random_stretch)

    def __len__(self):
        return len(self.video_folders)
    
    def print_dataset(self):
        for i in range(len(self)):
            print("[{}] {} {} {}".format(i, self.video_labels[i], self.video_folders[i], self.scales[i]))

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_label = self.video_labels[idx]
        scale = self.scales[idx]

        all_frames = [os.path.join(video_folder, frame_name) for frame_name in os.listdir(video_folder)]
        all_frames.sort()
        if len(all_frames) > 150:
            all_frames = all_frames[::scale]

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
        # trans = random.randint(0,1)
        # T = np.float32([[1,0,random.randint(-25,25)],[0,1,random.randint(-25,25)]])
        processed_frames = []
        for frame in all_frames:
            img = cv2.imread(frame)
            img = cv2.resize(img, (WIDTH, HEIGHT))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img, 1) if flip else img
            # img = cv2.warpAffine(img, T, (WIDTH, HEIGHT)) if trans else img
            processed_frames.append(img)

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
        
        noise = random.randint(0,1)
        if self.mode == "train" and noise:
            frames = frames + 0.1 * torch.randn(self.window_num*self.clip_num, 3, self.frame_num, WIDTH, HEIGHT)

        return frames, video_label

class FullKineticsDataset(Dataset):
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
        self.class_names = random.sample(all_class_names, self.class_num)
        self.labels = dict()
        for i, class_name in enumerate(self.class_names):
            self.labels[class_name] = i+1

        self.video_folders = []
        self.video_names = []
        self.video_labels = []
        self.scales = []
        for class_name in self.class_names:
            label = self.labels[class_name]
            class_folder = os.path.join(self.data_folder, class_name)
            video_names = self.get_video_names(class_folder)
            random.shuffle(video_names)
            video_names = video_names[:self.inst_num]

            for video_name in video_names:
                random_stretch = random.randint(1,3)
                self.video_folders.append(class_folder)
                self.video_names.append(video_name)
                self.video_labels.append(label)
                self.scales.append(random_stretch)
    
    def get_video_names(self, class_folder):
        video_names = []
        frame_names = os.listdir(class_folder)
        for frame_name in frame_names:
            frame_name, _ = os.path.splitext(frame_name)
            frame_name = frame_name[:-4]
            video_names.append(frame_name)
        return list(set(video_names))

    def __len__(self):
        return len(self.video_folders)
    
    def print_dataset(self):
        for i in range(len(self)):
            print("[{}] {} {} {}".format(i, self.video_labels[i], self.video_folders[i], self.scales[i]))

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_name = self.video_names[idx]
        video_label = self.video_labels[idx]
        scale = self.scales[idx]

        all_frames = []
        i = 1
        while True:
            frame_name = "0000" + str(i)
            frame_name = video_name + frame_name[-4:] + ".jpg"
            frame_path = os.path.join(video_folder, frame_name)
            if os.path.exists(frame_path):
                all_frames.append(frame_path)
                i += 1
            else:
                break
        all_frames.sort()
        all_frames = all_frames[::scale]

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
        # trans = random.randint(0,1)
        # T = np.float32([[1,0,random.randint(-25,25)],[0,1,random.randint(-25,25)]])
        processed_frames = []
        for frame in all_frames:
            img = cv2.imread(frame)
            img = cv2.resize(img, (WIDTH, HEIGHT))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img, 1) if flip else img
            # img = cv2.warpAffine(img, T, (WIDTH, HEIGHT)) if trans else img
            processed_frames.append(img)

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
        
        # noise = random.randint(0,1)
        # if self.mode == "train" and noise:
        #     frames = frames + 0.1 * torch.randn(self.window_num*self.clip_num, 3, self.frame_num, WIDTH, HEIGHT)

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
        self.class_names = random.sample(all_class_names, self.class_num)
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

class ClassBalancedSampler(Sampler):

    def __init__(self, num_per_class, num_cl, inst_num, shuffle):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.inst_num = inst_num
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        batch = []
        for j in range(self.num_cl):
            sublist = []
            for i in range(self.inst_num):
                sublist.append(i+j*self.inst_num)
            random.shuffle(sublist)
            sublist = sublist[:self.num_per_class]
            batch.append(sublist)

        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        
        return iter(batch)

    def __len__(self):
        return 1

def get_data_loader(dataset, num_per_class, shuffle=False):
    sampler = ClassBalancedSampler(num_per_class, dataset.class_num, dataset.inst_num, shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*dataset.class_num, sampler=sampler, num_workers=0)
    return loader

# TCN_OUT_CHANNEL = 64                        # Num of channels of output of TCN
# RELATION_DIM = 32                           # Dim of one layer of relation net
# CLASS_NUM = 3                               # <X>-way  | Num of classes
# SAMPLE_NUM = 5                              # <Y>-shot | Num of supports per class
# QUERY_NUM = 3                               # Num of instances for query per class
# TRAIN_EPISODE = 20000                       # Num of training episode 
# VALIDATION_EPISODE = 100                    # Num of validation episode
# VALIDATION_FREQUENCY = 200                  # After each <X> training episodes, do validation once
# LEARNING_RATE = 0.0001                      # Initial learning rate
# FRAME_NUM = 10                              # Num of frames per clip
# CLIP_NUM = 5                                # Num of clips per window
# WINDOW_NUM = 3                              # Num of processing window per video
# INST_NUM = 10                               # Num of videos selected in each class
# DATA_FOLDER = "/data/ssongad/kinetics400/frame/train"
# TRAIN_SPLIT = "/data/ssongad/kinetics400/train.txt"                  # For Kinetics dataset only
# TEST_SPLIT = "/data/ssongad/kinetics400/test.txt"                    #
# def read_split(file_path):
#     result = []
#     if os.path.exists(file_path):
#         file = open(file_path, "r")
#         lines = file.readlines()
#         file.close()

#         for line in lines:
#             result.append(line.rstrip())
            
#     return result

# TRAIN_SPLIT = read_split(TRAIN_SPLIT)
# TEST_SPLIT = read_split(TEST_SPLIT)
# the_dataset = FullKineticsDataset(DATA_FOLDER, "train", (TRAIN_SPLIT, TEST_SPLIT), CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
# print(the_dataset[1])