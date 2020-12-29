import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import os
import random
import argparse

from relation_net import RelationNetwork as RN
from encoder import Simple3DEncoder as C3D
from tcn import TemporalConvNet as TCN
from attention_pool import AttentionPooling as AP
import dataset
from utils import *

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="path of the dataset json file", required=True)
parser.add_argument("-w", "--way", help="number of classes", type=int, default=3)
parser.add_argument("-s", "--shot", help="number of shots", type=int, default=5)
parser.add_argument("-g", "--gpu", help="indices of gpu to be used, use all if not specified, e.g. --gpu=2,4,5")
parser.add_argument("-t", "--test_ep", help="number of test episodes", type=int, default=500)
parser.add_argument("-p", "--predict", help="whether to use mse or ctc at prediction", choices=["mse", "ctc"], default="ctc")
parser.add_argument("-c", "--checkpoint", help="path of a checkpoint to start from, a path with its name as the accuracy", required=True)

args = parser.parse_args()

if not os.exists(args.dataset):
    raise Exception("invalid dataset path: {}".format(args.dataset))
else:
    file = open(args.dataset, 'r')
    text = file.readline()
    file.close()
    dataset_info = json.loads(text)
if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda')
if args.way > 1:
    CLASS_NUM = args.way
else:
    raise Exception("one-way or even less is not meaningful")
if args.shot >= 1:
    SAMPLE_NUM = args.shot
else:
    raise Exception("zero-shot is beyond the scope of this project")
if not os.path.exists(args.checkpoint):
    raise Exception("invalid checkpoint path: {}".format(args.checkpoint))

# Some Constants
CLIP_NUM = 5    # Num of clips per window
WINDOW_NUM = 3  # Num of processing window per video
FRAME_NUM = 10  # Num of frames per clip
QUERY_NUM = 5   # Num of instances for query per class
INST_NUM = 10   # Num of videos selected in each class (A silly design, will be removed later)
TCN_OUT = 64    # Num of channels of output of TCN

# Define models
c3d = C3D(in_channels=3) 
c3d = nn.DataParallel(c3d)
tcn = TCN(245760, [128,128,64,TCN_OUT])
ap = AP(CLASS_NUM, SAMPLE_NUM, QUERY_NUM, WINDOW_NUM, CLIP_NUM, TCN_OUT)
rn = RN(CLIP_NUM, hidden_size=32) 

# Move models to GPU
c3d.to(device)
rn.to(device)
tcn.to(device)
ap.to(device)

# Load Saved Models & Optimizers & Schedulers
my_load(c3d, "c3d.pkl")
my_load(tcn, "tcn.pkl")
my_load(ap, "ap.pkl")
my_load(rn, "rn.pkl")

# Testing
with torch.no_grad():
    accuracies = []

    test_ep = 0
    while test_ep < args.test_ep:
        
        # Data Loading
        try:
            if dataset_info["name"] != "finegym":
                the_dataset = dataset.StandardDataset(dataset_info["folders"], "test", dataset_info["split"], CLASS_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
            else:
                the_dataset = dataset.FinegymDataset(dataset_info["folder"], dataset_info["finegym_info"], "test", dataset_info["split"], CLASS_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
            dataloader = dataset.get_data_loader(the_dataset, num_per_class=SAMPLE_NUM+QUERY_NUM, num_workers=0)
            data, data_labels = dataloader.__iter__().next()     # [class*(support+query), window*clip, RGB, frame, H, W]
        except Exception:
            continue
        data = data.view(-1, 3, FRAME_NUM, 128, 128)

        print("Test_Epi[{}]".format(test_ep), end="\t")

        # Generate support & query split
        query_index = []
        support_index = []
        for i in range(CLASS_NUM):
            start = (SAMPLE_NUM+QUERY_NUM) * i
            end = (SAMPLE_NUM+QUERY_NUM) * (i+1)
            index = list(range(start, end))
            q = random.sample(index, QUERY_NUM)
            s = list(set(index)-set(q))
            query_index.extend(q)
            support_index.extend(s)
        random.shuffle(query_index)
        query_index = torch.tensor(query_index)
        support_index = torch.tensor(support_index)
        
        # Encoding
        embed = c3d(Variable(data).to(device))
        embed = embed.view(CLASS_NUM*(SAMPLE_NUM+QUERY_NUM), WINDOW_NUM*CLIP_NUM, -1)  # [class*(support+query), window*clip, feature]

        # TCN Processing
        embed = torch.transpose(embed, 1, 2)           # [class*(support+query), feature(channel), window*clip(length)]
        embed = tcn(embed)
        embed = torch.transpose(embed, 1, 2)           # [class*(support+query), window*clip, feature]

        # Split data into support & query
        samples = embed[support_index] # [class*support, window*clip, feature]
        batches = embed[query_index]   # [class*query, window*clip, feature]
        batches_labels = data_labels[query_index]

        # Attention Pooling
        samples = samples.reshape(CLASS_NUM*SAMPLE_NUM*WINDOW_NUM, CLIP_NUM, -1)  # [class*sample*window, clip, feature]
        batches = batches.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM, CLIP_NUM, -1)  # [query*class*window, clip, feature]
        samples = ap(samples, batches)                    # [query*class*window, class, clip, feature]

        # Compute Relation
        batches_rn = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1)  # [class, query*class*window, clip, feature]
        batches_rn = torch.transpose(batches_rn,0,1)               # [query*class*window, class, clip, feature]
        relations = torch.cat((samples,batches_rn),2).reshape(-1,CLIP_NUM*2,TCN_OUT)    # [query*class*window, class, clip*2(channel), feature]
        relations = rn(relations).reshape(QUERY_NUM*CLASS_NUM, WINDOW_NUM, CLASS_NUM)    # [query*class, window, class]
        relations_ctc = torch.cat((blank_prob, relations), 2)             # [query*class, window(length), class+1]
        final_outcome = torch.transpose(logSoftmax(relations_ctc), 0, 1)  # [window(length), query*class, class+1]
        
        cos = nn.CosineSimilarity(dim=2)
        samples = samples.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM, CLASS_NUM, -1)
        batches = batches.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM,1,-1).repeat(1,CLASS_NUM,1)
        relations = cos(samples,batches).reshape(QUERY_NUM*CLASS_NUM, WINDOW_NUM, CLASS_NUM)
        
        # Generate final probabilities
        blank_prob = torch.full(size=(QUERY_NUM*CLASS_NUM, WINDOW_NUM, 1), fill_value=1, dtype=torch.float).to(device)
        relations_ctc = torch.cat((blank_prob, relations), 2)
        final_outcome = nn.functional.softmax(relations_ctc, 2)  # [query*class, window(length), class+1]

        # Predict
        batches_labels = batches_labels.numpy()
        if args.predict == "mse":
            relations_mse = nn.functional.softmax(torch.sum(relations, 1), dim=1) # [query*class, class]
            _, predict_labels = torch.max(relations_mse.data, 1)
            batches_labels = batches_labels - 1
        else:
            predict_labels = ctc_predict(final_outcome.cpu().numpy())
            predict_labels = ctc_predict_single(final_outcome)

        rewards = [1 if predict_labels[i] == batches_labels[i] else 0 for i in range(len(predict_labels))]
        total_rewards = np.sum(rewards)

        # Record accuracy
        accuracy = total_rewards/(CLASS_NUM * QUERY_NUM)
        accuracies.append(accuracy)
        print("Accuracy = {}".format(accuracy), end='\t')
        test_accuracy, _ = mean_confidence_interval(accuracies)
        print("Average Accuracy = {}".format(test_accuracy))

        test_ep += 1

# Average accuracy
test_accuracy, _ = mean_confidence_interval(accuracies)
print("Final_Accu = {}".format(test_accuracy))