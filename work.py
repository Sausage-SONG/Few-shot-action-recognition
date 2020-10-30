import torch                                     #
import torch.nn as nn                            #  Pytorch

import numpy as np                               #  Numpy
import os                                        #  OS

from config import *
import dataset                                      #  Dataset
                   #



# Setup Data
try:
    if DATASET == "haa":
        the_dataset = dataset.HAADataset(DATA_FOLDERS, None, "train", CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
    elif DATASET == "kinetics":
        the_dataset = dataset.KineticsDataset(DATA_FOLDER, "train", (TRAIN_SPLIT, TEST_SPLIT), CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
    elif DATASET == "full_kinetics":
        the_dataset = dataset.FullKineticsDataset(DATA_FOLDER, "train", (TRAIN_SPLIT, TEST_SPLIT), CLASS_NUM, SAMPLE_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
    sample_dataloader = dataset.get_data_loader(the_dataset,num_per_class=SAMPLE_NUM)
    batch_dataloader = dataset.get_data_loader(the_dataset,num_per_class=QUERY_NUM,shuffle=True)
    samples, _ = sample_dataloader.__iter__().next()             # [support*class, window*clip, RGB, frame, H, W]
    batches, batches_labels = batch_dataloader.__iter__().next()   # [query*class, window*clip, RGB, frame, H, W]
except Exception:
    print("skiped")

print(torch.std(samples))
print(torch.std(batches))
