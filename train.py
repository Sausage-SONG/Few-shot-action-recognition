import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

import numpy as np
import os
import random
import argparse
import json

from relation_net import RelationNetwork as RN
from encoder import Simple3DEncoder as C3D
from tcn import TemporalConvNet as TCN
from attention_pool import AttentionPooling as AP
import dataset
from utils import *

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="path of the dataset json file", required=True)
parser.add_argument("-n", "--exp_name", help="experiment name, this determines where to save trained models", required=True)
parser.add_argument("-w", "--way", help="number of classes", type=int, default=3)
parser.add_argument("-s", "--shot", help="number of shots", type=int, default=5)
parser.add_argument("-g", "--gpu", help="indices of gpu to be used, use all if not specified, e.g. --gpu=2,4,5")
parser.add_argument("-t", "--train_ep", help="number of training episodes", type=int, default=30000)
parser.add_argument("-v", "--valid_ep", help="number of validation episodes", type=int, default=50)
parser.add_argument("-f", "--valid_frq", help="validation frequency, a number of training episodes", type=int, default=200)
parser.add_argument("-l", "--lr", help="learning rate", type=float, default=0.001)
parser.add_argument("--load_frq", help="loading frequency, one means load new data each episode", type=int, default=3)
parser.add_argument("-m", "--mse_also", help="whether to use mse together with ctc loss", action="store_true")
parser.add_argument("-p", "--predict", help="whether to use mse or ctc at prediction", choices=["mse", "ctc"], default="ctc")
parser.add_argument("-c", "--checkpoint", help="path of a checkpoint to start from, a path with its name as the accuracy")

args = parser.parse_args()

if not os.exists(args.dataset):
    raise Exception("invalid dataset path: {}".format(args.dataset))
else:
    file = open(args.dataset, 'r')
    text = file.readline()
    file.close()
    dataset_info = json.loads(text)
if args.load_frq <= 0:
    raise Exception("loading frequency must be positive")
if args.train_ep <= 0 or args.valid_ep <= 0:
    raise Exception("training and validation episodes must be positive")
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
if args.checkpoint is not None and not os.path.exists(args.checkpoint):
    raise Exception("invalid checkpoint path: {}".format(args.checkpoint))

# Some Constants
CLIP_NUM = 5    # Num of clips per window
WINDOW_NUM = 3  # Num of processing window per video
FRAME_NUM = 10  # Num of frames per clip
QUERY_NUM = 5   # Num of instances for query per class
INST_NUM = 10   # Num of videos selected in each class (A silly design, will be removed later)
TCN_OUT = 64    # Num of channels of output of TCN
max_accuracy = 0

# Define Models
c3d = C3D(in_channels=3) 
c3d = nn.DataParallel(c3d)
tcn = TCN(245760, [128,128,64,TCN_OUT])
ap = AP(CLASS_NUM, SAMPLE_NUM, QUERY_NUM, WINDOW_NUM, CLIP_NUM, TCN_OUT)
rn = RN(CLIP_NUM, hidden_size=32) 

ctc = nn.CTCLoss()
logSoftmax = nn.LogSoftmax(2)
mse = nn.MSELoss()   

# Move models to computing device
c3d.to(device)
tcn.to(device)
ap.to(device)
rn.to(device)

logSoftmax.to(device)
ctc.to(device)
mse.to(device)

# Define Optimizers
c3d_optim = torch.optim.AdamW(c3d.parameters(), lr=args.lr)
rn_optim = torch.optim.AdamW(rn.parameters(), lr=args.lr)
tcn_optim = torch.optim.AdamW(tcn.parameters(), lr=args.lr)
ap_optim = torch.optim.AdamW(ap.parameters(), lr=args.lr)

# Define Schedulers
c3d_scheduler = StepLR(c3d_optim, step_size=3000, gamma=0.5)
rn_scheduler = StepLR(rn_optim, step_size=3000, gamma=0.5)
tcn_scheduler = StepLR(tcn_optim, step_size=3000, gamma=0.5)
ap_scheduler = StepLR(ap_optim, step_size=3000, gamma=0.5)

# Load Saved Models & Optimizers & Schedulers
if args.checkpoint is not None:
    my_load(c3d, "c3d.pkl")
    my_load(tcn, "tcn.pkl")
    my_load(ap, "ap.pkl")
    my_load(rn, "rn.pkl")
    my_load(c3d_optim, "c3d_optim.pkl")
    my_load(tcn_optim, "tcn_optim.pkl")
    my_load(ap_optim, "ap_optim.pkl")
    my_load(rn_optim, "rn_optim.pkl")
    my_load(c3d_scheduler, "c3d_scheduler.pkl")
    my_load(tcn_scheduler, "tcn_scheduler.pkl")
    my_load(ap_scheduler, "ap_scheduler.pkl")
    my_load(rn_scheduler, "rn_scheduler.pkl")

    tmp = os.path.split(args.checkpoint)[1]
    if "Latest_" in tmp:
        tmp = tmp[7:]
    max_accuracy = float(tmp)

# Prepare output folder
output_folder = os.path.join("./models", args.exp_name)
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Some Constant Tensors
input_lengths = torch.full(size=(QUERY_NUM*CLASS_NUM,), fill_value=WINDOW_NUM, dtype=torch.long).to(device)
target_lengths = torch.full(size=(QUERY_NUM*CLASS_NUM,), fill_value=1, dtype=torch.long).to(device)
blank_prob = torch.full(size=(QUERY_NUM*CLASS_NUM, WINDOW_NUM, 1), fill_value=1, dtype=torch.float).to(device)

# Training Loop
train_ep = 0
while train_ep < args.train_ep:

    # Load Data
    if train_ep % args.load_frq == 0:
        try:
            if dataset_info["name"] != "finegym":
                the_dataset = dataset.StandardDataset(dataset_info["folders"], "train", dataset_info["split"], CLASS_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
            else:
                the_dataset = dataset.FinegymDataset(dataset_info["folder"], dataset_info["finegym_info"], "train", dataset_info["split"], CLASS_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
            dataloader = dataset.get_data_loader(the_dataset, num_per_class=SAMPLE_NUM+QUERY_NUM, num_workers=0)
            data, data_labels = dataloader.__iter__().next()     # [class*(support+query), window*clip, RGB, frame, H, W]
        except Exception:
            continue
        data = data.view(-1, 3, FRAME_NUM, 128, 128)
    
    print("Train_Ep[{}] Current_Accuracy = {}".format(train_ep, max_accuracy), end="\t")
    
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

    if args.mse_also:
        relations_mse = nn.functional.softmax(torch.sum(relations, 1), dim=1) # [query*class, class]
        one_hot_labels = Variable(torch.zeros(QUERY_NUM*CLASS_NUM, CLASS_NUM).scatter_(1, (batches_labels-1).view(-1,1), 1).to(device))
        loss = mse(relations_mse, one_hot_labels) + ctc(final_outcome, batches_labels, input_lengths, target_lengths)
    else:
        loss = ctc(final_outcome, batches_labels, input_lengths, target_lengths)
    print("Loss = {}".format(loss))

    # Back Propagation
    c3d.zero_grad()
    tcn.zero_grad()
    ap.zero_grad()
    rn.zero_grad()
    loss.backward()

    # Clip Gradient
    nn.utils.clip_grad_norm_(c3d.parameters(),0.5)
    nn.utils.clip_grad_norm_(tcn.parameters(),0.5)
    nn.utils.clip_grad_norm_(ap.parameters(),0.5)
    nn.utils.clip_grad_norm_(rn.parameters(),0.5)

    # Update Models
    c3d_optim.step()
    tcn_optim.step()
    rn_optim.step()
    ap_optim.step()

    # Update "step" for scheduler
    c3d_scheduler.step()
    tcn_scheduler.step()
    ap_scheduler.step()
    rn_scheduler.step()

    train_ep += 1

    # Validation Loop
    if (train_ep % args.valid_frq == 0 and train_ep != 0) or train_ep == args.train_ep:

        with torch.no_grad():
            accuracies = []

            valid_ep = 0
            while valid_ep < args.valid_ep:

                # Data Loading
                try:
                    if DATASET in ['haa', 'mit']:
                        the_dataset = dataset.StandardDataset(DATA_FOLDERS, "test", (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT), CLASS_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
                    elif DATASET in ['finegym']:
                        the_dataset = dataset.FinegymDataset(DATA_FOLDERS, INFO_DICT, "test", [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT], CLASS_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
                    sample_dataloader = dataset.get_data_loader(the_dataset, num_per_class=SAMPLE_NUM, num_workers=0)
                    batch_dataloader = dataset.get_data_loader(the_dataset, num_per_class=QUERY_NUM,shuffle=True, num_workers=0)
                    samples, _ = sample_dataloader.__iter__().next()            # [query*class, clip, RGB, frame, H, W]
                    batches, batches_labels = batch_dataloader.__iter__().next()   # [query*class, window*clip, RGB, frame, H, W]
                except Exception:
                    continue
                
                print("Val_Ep[{}] Pres_Accu = {}".format(valid_ep, max_accuracy), end="\t")

                # Encoding
                samples = samples.view(CLASS_NUM*SAMPLE_NUM*WINDOW_NUM*CLIP_NUM, 3, FRAME_NUM, 128, 128)
                samples = c3d(Variable(samples).to(device))
                samples = samples.view(CLASS_NUM*SAMPLE_NUM, WINDOW_NUM*CLIP_NUM, -1)    # [support*class, window*clip, feature]

                batches = batches.view(CLASS_NUM*QUERY_NUM*WINDOW_NUM*CLIP_NUM, 3, FRAME_NUM, 128, 128)
                batches = c3d(Variable(batches).to(device))
                batches = batches.view(CLASS_NUM*QUERY_NUM, WINDOW_NUM*CLIP_NUM,-1)       # [query*class, window*clip, feature]

                # TCN Processing
                samples = torch.transpose(samples,1,2)       # [support*class, feature(channel), window*clip(length)]
                samples = tcn(samples)
                samples = torch.transpose(samples,1,2)       # [support*class, window*clip, feature]
                samples = samples.reshape(CLASS_NUM*SAMPLE_NUM*WINDOW_NUM, CLIP_NUM, -1)  # [class*sample*window, clip, feature]

                batches = torch.transpose(batches,1,2)       # [query*class, feature(channel), window*clip(length)]
                batches = tcn(batches)
                batches = torch.transpose(batches,1,2)       # [query*class, window*clip, feature]
                batches = batches.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM, CLIP_NUM, -1)  # [query*class*window, clip, feature]

                # Attention Pooling
                samples = ap(samples, batches)                    # [query*class*window, class, clip, feature]

                # Compute Relation
                batches_rn = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1)  # [class, query*class*window, clip, feature]
                batches_rn = torch.transpose(batches_rn,0,1)               # [query*class*window, class, clip, feature]
                relations = torch.cat((samples,batches_rn),2).reshape(-1,CLIP_NUM*2,TCN_OUT)    # [query*class*window, class, clip*2(channel), feature]
                relations = rn(relations).reshape(QUERY_NUM*CLASS_NUM, WINDOW_NUM, CLASS_NUM)    # [query*class, window, class]

                # Generate final probabilities
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
                print("Accuracy = {}".format(accuracy))

                valid_ep += 1

            # Average accuracy
            val_accuracy, _ = mean_confidence_interval(accuracies)
            print("Average Val_Accu = {}".format(val_accuracy))

            # Save Model
            if val_accuracy > max_accuracy:
                # Prepare folder
                folder_for_this_accuracy = os.path.join(output_folder, str(val_accuracy))
                max_accuracy = val_accuracy
                print("Models Saved with accuracy={}".format(max_accuracy))
            else:
                folder_for_this_accuracy = os.path.join(output_folder, "Latest_{}".format(val_accuracy))

            if not os.path.exists(folder_for_this_accuracy):
                os.mkdir(folder_for_this_accuracy)

            # Save networks
            torch.save(c3d.state_dict(), os.path.join(folder_for_this_accuracy, "c3d.pkl"))
            torch.save(rn.state_dict(), os.path.join(folder_for_this_accuracy, "rn.pkl"))
            torch.save(tcn.state_dict(), os.path.join(folder_for_this_accuracy, "tcn.pkl"))
            torch.save(ap.state_dict(), os.path.join(folder_for_this_accuracy, "ap.pkl"))

            torch.save(c3d_optim.state_dict(), os.path.join(folder_for_this_accuracy, "c3d_optim.pkl"))
            torch.save(rn_optim.state_dict(), os.path.join(folder_for_this_accuracy, "rn_optim.pkl"))
            torch.save(tcn_optim.state_dict(), os.path.join(folder_for_this_accuracy, "tcn_optim.pkl"))
            torch.save(ap_optim.state_dict(), os.path.join(folder_for_this_accuracy, "ap_optim.pkl"))

            torch.save(c3d_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "c3d_scheduler.pkl"))
            torch.save(rn_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "rn_scheduler.pkl"))
            torch.save(tcn_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "tcn_scheduler.pkl"))
            torch.save(ap_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "ap_scheduler.pkl"))

print("Training Done")
print("Final Accuracy = {}".format(max_accuracy))
