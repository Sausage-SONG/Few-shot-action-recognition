# Public Packages
import torch                                     #
import torch.nn as nn                            #  Pytorch
from torch.autograd import Variable              #

import numpy as np                               #  Numpy
import os                                        #  OS

# Local Packages
from relationNet import RelationNetwork as RN       #  Relation Network
from relationNet import RelationNetworkZero as RN0  #  RN0
from encoder import Simple3DEncoder as C3D          #  Conv3D
from tcn import TemporalConvNet as TCN              #  TCN
from attention_pool import AttentionPooling as AP   #  Attention Pooling
import dataset                                      #  Dataset
from utils import *                                 #  Helper Functions

# Constant (Settings)
TCN_OUT_CHANNEL = 64                    # Num of channels of output of TCN
RELATION_DIM = 32                       # Dim of one layer of relation net
CLASS_NUM = 3                           # <X>-way  | Num of classes
SAMPLE_NUM = 5                          # <Y>-shot | Num of supports per class
QUERY_NUM = 3                           # Num of instances for validation per class
TEST_EPISODE = 500                      # Num of validation episode
FRAME_NUM = 10                          # Num of frames per clip
CLIP_NUM = 5                            # Num of clips per window
WINDOW_NUM = 3                          # Num of processing window per video
INST_NUM = 15                           # Num of videos selected in each class

DATA_FOLDER = "/data/jchungaa/finegym/frame"

encoder_saved_model = "/data/ssongad/codes/ctc_ap_v2/models/CTC_Full_Finegym_Unsupervised_Attention_v2_5S/Latest/encoder.pkl"     # Path of saved encoder model
tcn_saved_model = "/data/ssongad/codes/ctc_ap_v2/models/CTC_Full_Finegym_Unsupervised_Attention_v2_5S/Latest/tcn.pkl"             # Path of saved tcn model
ap_saved_model = "/data/ssongad/codes/ctc_ap_v2/models/CTC_Full_Finegym_Unsupervised_Attention_v2_5S/Latest/ap.pkl"  
rn_saved_model = "/data/ssongad/codes/ctc_ap_v2/models/CTC_Full_Finegym_Unsupervised_Attention_v2_5S/Latest/rn.pkl"               # Path of saved relation net model
rn0_saved_model = "/data/ssongad/codes/ctc_ap_v2/models/CTC_Full_Finegym_Unsupervised_Attention_v2_5S/Latest/rn0.pkl"  

# Device to be used
os.environ['CUDA_VISIBLE_DEVICES']="8"                          # GPU to be used
device = torch.device('cuda')

TEST_SPLIT = "/data/ssongad/finegym/test.txt"
TEST_SPLIT = read_split(TEST_SPLIT)
TEST_SPLIT = [str(i) for i in range(288)]

INFO_DICT = ["/data/ssongad/finegym/gym288_train_element_v1.1.txt", "/data/ssongad/finegym/gym288_val_element.txt"]
d = dict()
for path in INFO_DICT:
    file = open(path, 'r')
    lines = file.readlines()
    file.close()

    for line in lines:
        line = line.strip()
        line = line.split(" ")
        label = line[1]
        name = line[0]

        if label in d.keys():
            d[label].append(name)
        else:
            d[label] = [name]
INFO_DICT = d

def main():

    # Define models
    encoder = C3D(in_channels=3)
    encoder = nn.DataParallel(encoder)
    rn = RN(CLIP_NUM, RELATION_DIM)
    rn0 = RN0(CLIP_NUM*(CLASS_NUM+1), RELATION_DIM)
    tcn = TCN(245760, [128,128,64,TCN_OUT_CHANNEL])
    ap = AP(CLASS_NUM, SAMPLE_NUM, QUERY_NUM, WINDOW_NUM, CLIP_NUM, TCN_OUT_CHANNEL)

    # Move models to GPU
    encoder.to(device)
    rn.to(device)
    rn0.to(device)
    tcn.to(device)
    ap.to(device)

    # Load Saved Model
    if os.path.exists(encoder_saved_model):
        encoder.load_state_dict(torch.load(encoder_saved_model))
    if os.path.exists(rn_saved_model):
        rn.load_state_dict(torch.load(rn_saved_model))
    if os.path.exists(tcn_saved_model):
        tcn.load_state_dict(torch.load(tcn_saved_model))
    if os.path.exists(rn0_saved_model):
        rn0.load_state_dict(torch.load(rn0_saved_model))
    if os.path.exists(ap_saved_model):
        ap.load_state_dict(torch.load(ap_saved_model))

    # per_class_correct = dict()
    # per_class_count = dict()

    # Testing
    with torch.no_grad():
        accuracies = []

        test_episode = 0
        # while test_episode < TEST_EPISODE:
        while test_episode < TEST_EPISODE:
            print("Test_Epi[{}]".format(test_episode), end="\t")
            
            # Data Loading
            try:
                the_dataset = dataset.FinegymDataset(DATA_FOLDER, INFO_DICT, "test", [None, None, TEST_SPLIT], CLASS_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
                sample_dataloader = dataset.get_data_loader(the_dataset, num_per_class=SAMPLE_NUM)
                batch_dataloader = dataset.get_data_loader(the_dataset, num_per_class=QUERY_NUM, shuffle=True)
                samples, samples_labels = sample_dataloader.__iter__().next()            # [batch, clip, RGB, frame, H, W]
                batches, batches_labels = batch_dataloader.__iter__().next()   # [batch, window*clip, RGB, frame, H, W]
            except Exception:
                print("Skipped")
                continue

            total_covered = CLASS_NUM * QUERY_NUM
            
            # Encoding
            samples = samples.view(CLASS_NUM*SAMPLE_NUM*WINDOW_NUM*CLIP_NUM, 3, FRAME_NUM, 128, 128)
            samples = encoder(Variable(samples).to(device))
            samples = samples.view(CLASS_NUM*SAMPLE_NUM, WINDOW_NUM*CLIP_NUM, -1)    # [support*class, window*clip, feature]

            batches = batches.view(CLASS_NUM*QUERY_NUM*WINDOW_NUM*CLIP_NUM, 3, FRAME_NUM, 128, 128)
            batches = encoder(Variable(batches).to(device))
            batches = batches.view(CLASS_NUM*QUERY_NUM, WINDOW_NUM*CLIP_NUM,-1)       # [query*class, window*clip, feature]

            # TCN Processing
            samples = torch.transpose(samples,1,2)       # [support*class, feature(channel), window*clip(length)]
            samples = tcn(samples)
            samples = torch.transpose(samples,1,2)       # [support*class, window*clip, feature]
            samples = samples.reshape(CLASS_NUM*SAMPLE_NUM*WINDOW_NUM, CLIP_NUM, -1)  # [class*sample*window, clip, feature]
            # samples, _ = torch.max(samples, 2)           # [class, sample, clip, feature]
            # samples = torch.mean(samples,1)              # [class, clip, feature]

            batches = torch.transpose(batches,1,2)       # [query*class, feature(channel), window*clip(length)]
            batches = tcn(batches)
            batches = torch.transpose(batches,1,2)       # [query*class, window*clip, feature]
            batches = batches.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM, CLIP_NUM, -1)  # [query*class*window, clip, feature]

            # Attention Pooling
            # before = samples.reshape(CLASS_NUM*SAMPLE_NUM*WINDOW_NUM, -1).cpu().numpy()
            samples = ap(samples, batches)
            # after = samples.reshape(QUERY_NUM*CLASS_NUM*WINDOW_NUM*CLASS_NUM, -1).cpu().numpy()
            # batches_rn = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1)  # [class, query*class*window, clip, feature]
            # batches_rn = torch.transpose(batches_rn,0,1)               # [query*class*window, class, clip, feature]
            # query = batches_rn.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM*CLASS_NUM, -1).cpu().numpy()

            # file = open("tmp.txt", "w")
            # file.write("dataset\n"+the_dataset.print_dataset()+"\n")
            # file.write("batches\n"+str(batches_labels.shape)+"\n"+str(batches_labels)+"\n")
            # for i in range(9):
            #     file.write(batches_folders[i]+"\n")
            # file.write("samples\n"+str(samples_labels.shape)+"\n"+str(samples_labels)+"\n")
            # for i in range(15):
            #     file.write(samples_folders[i]+"\n")
            # file.write("weight\n"+str(weight.shape)+"\n")
            # for i in range(81):
            #     for j in range(15):
            #         file.write("{:.4f}".format(float(weight[i][0][j]))+" ")
            #     file.write("\n")
            # file.close()
            # return

            # import matplotlib.pyplot as plt
            # plt.imshow(before, cmap='hot', interpolation='nearest')
            # plt.savefig('before.png')
            # plt.imshow(after, cmap='hot', interpolation='nearest')
            # plt.savefig('after.png')
            # plt.imshow(query, cmap='hot', interpolation='nearest')
            # plt.savefig('query.png')
            # return

            # Compute Relation
            batches_rn = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1)  # [class, query*class*window, clip, feature]
            batches_rn = torch.transpose(batches_rn,0,1)               # [query*class*window, class, clip, feature]
            relations = torch.cat((samples,batches_rn),2).reshape(-1,CLIP_NUM*2,TCN_OUT_CHANNEL)    # [query*class*window, class, clip*2(channel), feature]
            relations = rn(relations).reshape(QUERY_NUM*CLASS_NUM, WINDOW_NUM, CLASS_NUM)    # [query*class, window, class]

            # Compute Zero Probability
            samples_rn0 = samples.reshape(QUERY_NUM*CLASS_NUM*WINDOW_NUM, CLASS_NUM*CLIP_NUM, TCN_OUT_CHANNEL)  # [query*class*window, class*clip, feature]
            relations_rn0 = torch.cat((batches, samples_rn0), 1)   # [query*class*window, (class+1)*clip, feature]
            blank_prob = rn0(relations_rn0).reshape(QUERY_NUM*CLASS_NUM, WINDOW_NUM, 1)  # [query*class, window, 1]

            # Generate final probabilities
            relations = torch.cat((blank_prob, relations), 2)
            final_outcome = nn.functional.softmax(relations, 2)  # [query*class, window(length), class+1]

            # Predict
            # predict_labels = ctc_predict(final_outcome.cpu().numpy())
            predict_labels = ctc_predict_single(final_outcome)
            batches_labels = batches_labels.numpy()

            # Counting Correct Ones
            rewards = [compute_score(prediction, truth) for prediction, truth in zip(predict_labels, batches_labels)]
            total_rewards = np.sum(rewards)

            # # Per Class Record
            # class_labels = the_dataset.get_labels()
            # for i, reward in enumerate(rewards):
            #     label = batches_labels[i]
            #     for class_name in class_labels.keys():
            #         if class_labels[class_name] == label:
            #             if reward:
            #                 if class_name in per_class_correct.keys():
            #                     per_class_correct[class_name] += 1
            #                 else:
            #                     per_class_correct[class_name] = 1
            #             if class_name in per_class_count.keys():
            #                 per_class_count[class_name] += 1
            #             else:
            #                 per_class_count[class_name] = 1
                        
            # Record accuracy
            accuracy = total_rewards/total_covered
            accuracies.append(accuracy)
            test_accuracy, _ = mean_confidence_interval(accuracies)
            print("Accuracy = {}".format(accuracy))
            print(batches_labels)
            print(predict_labels)
            print(test_accuracy)
            print()

            test_episode += 1

        # Overall accuracy
        test_accuracy, _ = mean_confidence_interval(accuracies)
        print("Final_Accu = {}".format(test_accuracy))
        
        # file = open("test_log.txt", "a")
        # file.write(str(SAMPLE_NUM)+" "+str(test_accuracy)+"\n")
        # file.close()

        # file = open("test_log.txt", "w")
        # for key in per_class_count.keys():
        #     line = "{}, {}, {}, {}".format(key, per_class_count[key], per_class_correct[key], per_class_correct[key]/per_class_count[key])
        #     file.write(line+"\n")
        # file.close()
        

# Program Starts
if __name__ == '__main__':
    main()