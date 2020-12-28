import torch                                     #
import torch.nn as nn                            #  Pytorch
from torch.optim.lr_scheduler import StepLR      #
from torch.autograd import Variable              #

import numpy as np                               #  Numpy
import os                                        #  OS
import random                                    #  Random

from relationNet import RelationNetwork as RN       #  Relation Net
from relationNet import RelationNetworkZero as RN0  #
from encoder import Simple3DEncoder as C3D          #  Conv3D
from tcn import TemporalConvNet as TCN              #  TCN
from attention_pool import AttentionPooling as AP   #  Attention Pooling
import dataset                                      #  Dataset
from utils import *                                 #  Helper Functions
from config import *                                #  Config

# Device to be used
os.environ['CUDA_VISIBLE_DEVICES'] = GPU            # GPU to be used
device = torch.device('cuda')                       #

def main():
    # Define Models
    encoder = C3D(in_channels=3) 
    tcn = TCN(245760, [128,128,64,TCN_OUT_CHANNEL])  
    ap = AP(CLASS_NUM, SAMPLE_NUM, QUERY_NUM, WINDOW_NUM, CLIP_NUM, TCN_OUT_CHANNEL)
    rn = RN(CLIP_NUM, RELATION_DIM) 
    
    encoder = nn.DataParallel(encoder)

    ctc = nn.CTCLoss()
    logSoftmax = nn.LogSoftmax(2)
    mse = nn.MSELoss()   

    # Move models to computing device
    encoder.to(device)
    tcn.to(device)
    ap.to(device)
    rn.to(device)

    logSoftmax.to(device)
    ctc.to(device)
    mse.to(device)
    cos = nn.CosineSimilarity(dim=2).to(device)

    # Define Optimizers
    encoder_optim = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE)
    rn_optim = torch.optim.AdamW(rn.parameters(), lr=LEARNING_RATE)
    tcn_optim = torch.optim.AdamW(tcn.parameters(), lr=LEARNING_RATE)
    ap_optim = torch.optim.AdamW(ap.parameters(), lr=LEARNING_RATE)

    # Define Schedulers
    encoder_scheduler = StepLR(encoder_optim, step_size=3000, gamma=0.5)
    rn_scheduler = StepLR(rn_optim, step_size=3000, gamma=0.5)
    tcn_scheduler = StepLR(tcn_optim, step_size=3000, gamma=0.5)
    ap_scheduler = StepLR(ap_optim, step_size=3000, gamma=0.5)

    # Load Saved Models & Optimizers & Schedulers
    if os.path.exists(ENCODER_MODEL):
        encoder.load_state_dict(torch.load(ENCODER_MODEL))
    if os.path.exists(TCN_MODEL):
        tcn.load_state_dict(torch.load(TCN_MODEL))
    if os.path.exists(AP_MODEL):
        ap.load_state_dict(torch.load(AP_MODEL))
    if os.path.exists(RN_MODEL):
        rn.load_state_dict(torch.load(RN_MODEL))
    
    if os.path.exists(ENCODER_OPTIM):
        encoder_optim.load_state_dict(torch.load(ENCODER_OPTIM))
    if os.path.exists(RN_OPTIM):
        rn_optim.load_state_dict(torch.load(RN_OPTIM))
    if os.path.exists(TCN_OPTIM):
        tcn_optim.load_state_dict(torch.load(TCN_OPTIM))
    if os.path.exists(AP_OPTIM):
        ap_optim.load_state_dict(torch.load(AP_OPTIM))
    
    if os.path.exists(ENCODER_SCHEDULER):
        encoder_scheduler.load_state_dict(torch.load(ENCODER_SCHEDULER))
    if os.path.exists(RN_SCHEDULER):
        rn_scheduler.load_state_dict(torch.load(RN_SCHEDULER))
    if os.path.exists(TCN_SCHEDULER):
        tcn_scheduler.load_state_dict(torch.load(TCN_SCHEDULER))
    if os.path.exists(AP_SCHEDULER):
        ap_scheduler.load_state_dict(torch.load(AP_SCHEDULER))

    max_accuracy = MAX_ACCURACY     # Currently the best accuracy

    # Prepare output folder
    output_folder = os.path.join("./models", EXP_NAME)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # Some Constant Tensors
    input_lengths = torch.full(size=(QUERY_NUM*CLASS_NUM,), fill_value=WINDOW_NUM, dtype=torch.long).to(device)
    target_lengths = torch.full(size=(QUERY_NUM*CLASS_NUM,), fill_value=1, dtype=torch.long).to(device)
    blank_prob = torch.full(size=(QUERY_NUM*CLASS_NUM, WINDOW_NUM, 1), fill_value=1, dtype=torch.float).to(device)
    # zeros = torch.zeros(QUERY_NUM*CLASS_NUM, WINDOW_NUM).to(device)

    skipped = 0

    # Training Loop
    episode = 0
    while episode < TRAIN_EPISODE:

        print("Train_Epi[{}|{}] Pres_Accu = {}".format(episode, skipped, max_accuracy), end="\t")

        # Load Data
        if episode % TRAIN_FREQUENCY == 0:
            try:
                if DATASET in ["haa", 'mit']:
                    the_dataset = dataset.StandardDataset(DATA_FOLDERS, "train", (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT), CLASS_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
                elif DATASET == 'finegym':
                    the_dataset = dataset.FinegymDataset(DATA_FOLDERS, INFO_DICT, "train", [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT], CLASS_NUM, INST_NUM, FRAME_NUM, CLIP_NUM, WINDOW_NUM)
                dataloader = dataset.get_data_loader(the_dataset, num_per_class=SAMPLE_NUM+QUERY_NUM, num_workers=0)
                data, data_labels = dataloader.__iter__().next()     # [class*(support+query), window*clip, RGB, frame, H, W]
            except Exception:
                skipped += 1
                print("Skipped")
                continue
            data = data.view(-1, 3, FRAME_NUM, 128, 128)
        
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
        embed = encoder(Variable(data).to(device))
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
        relations = torch.cat((samples,batches_rn),2).reshape(-1,CLIP_NUM*2,TCN_OUT_CHANNEL)    # [query*class*window, class, clip*2(channel), feature]
        relations = rn(relations).reshape(QUERY_NUM*CLASS_NUM, WINDOW_NUM, CLASS_NUM)    # [query*class, window, class]

        # Generate final probabilities
        relations_ctc = torch.cat((blank_prob, relations), 2)             # [query*class, window(length), class+1]
        final_outcome = torch.transpose(logSoftmax(relations_ctc), 0, 1)  # [window(length), query*class, class+1]

        # Compute Loss
        relations_mse = nn.functional.softmax(torch.sum(relations, 1), dim=1) # [query*class, class]
        one_hot_labels = Variable(torch.zeros(QUERY_NUM*CLASS_NUM, CLASS_NUM).scatter_(1, (batches_labels-1).view(-1,1), 1).to(device))
        loss = mse(relations_mse, one_hot_labels) + ctc(final_outcome, batches_labels, input_lengths, target_lengths)
        # loss = ctc(final_outcome, batches_labels, input_lengths, target_lengths)
        print("Loss = {}".format(loss))

        # Back Propagation
        encoder.zero_grad()
        tcn.zero_grad()
        ap.zero_grad()
        rn.zero_grad()
        loss.backward()

        # Clip Gradient
        nn.utils.clip_grad_norm_(encoder.parameters(),0.5)
        nn.utils.clip_grad_norm_(tcn.parameters(),0.5)
        nn.utils.clip_grad_norm_(ap.parameters(),0.5)
        nn.utils.clip_grad_norm_(rn.parameters(),0.5)

        # Update Models
        encoder_optim.step()
        tcn_optim.step()
        rn_optim.step()
        ap_optim.step()

        # Update "step" for scheduler
        encoder_scheduler.step()
        tcn_scheduler.step()
        ap_scheduler.step()
        rn_scheduler.step()

        episode += 1

        # Validation Loop
        if (episode % VALIDATION_FREQUENCY == 0 and episode != 0) or episode == TRAIN_EPISODE:

            with torch.no_grad():
                accuracies = []

                validation_episode = 0
                while validation_episode < VALIDATION_EPISODE:

                    print("Val_Epi[{}] Pres_Accu = {}".format(validation_episode, max_accuracy), end="\t")

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
                        print("Skipped")
                        continue
                    
                    total_num_covered = CLASS_NUM * QUERY_NUM

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

                    batches = torch.transpose(batches,1,2)       # [query*class, feature(channel), window*clip(length)]
                    batches = tcn(batches)
                    batches = torch.transpose(batches,1,2)       # [query*class, window*clip, feature]
                    batches = batches.reshape(CLASS_NUM*QUERY_NUM*WINDOW_NUM, CLIP_NUM, -1)  # [query*class*window, clip, feature]

                    # Attention Pooling
                    samples = ap(samples, batches)                    # [query*class*window, class, clip, feature]

                    # Compute Relation
                    batches_rn = batches.unsqueeze(0).repeat(CLASS_NUM,1,1,1)  # [class, query*class*window, clip, feature]
                    batches_rn = torch.transpose(batches_rn,0,1)               # [query*class*window, class, clip, feature]
                    relations = torch.cat((samples,batches_rn),2).reshape(-1,CLIP_NUM*2,TCN_OUT_CHANNEL)    # [query*class*window, class, clip*2(channel), feature]
                    relations = rn(relations).reshape(QUERY_NUM*CLASS_NUM, WINDOW_NUM, CLASS_NUM)    # [query*class, window, class]

                    # Generate final probabilities
                    relations_ctc = torch.cat((blank_prob, relations), 2)
                    final_outcome = nn.functional.softmax(relations_ctc, 2)  # [query*class, window(length), class+1]

                    # Predict
                    relations_mse = nn.functional.softmax(torch.sum(relations, 1), dim=1) # [query*class, class]
                    _, predict_labels = torch.max(relations_mse.data, 1)
                    batches_labels = batches_labels - 1
                    # predict_labels = ctc_predict(final_outcome.cpu().numpy())
                    # predict_labels = ctc_predict_single(final_outcome)
                    batches_labels = batches_labels.numpy()

                    # Counting Correct Ones use ctc as score
                    # rewards = [compute_score(prediction, truth) for prediction, truth in zip(predict_labels, batches_labels)]
                    rewards = [1 if predict_labels[i] == batches_labels[i] else 0 for i in range(len(predict_labels))]
                    total_rewards = np.sum(rewards)

                    # Record accuracy
                    accuracy = total_rewards/total_num_covered
                    accuracies.append(accuracy)
                    print("Accu = {}".format(accuracy))

                    validation_episode += 1

                # Overall accuracy
                val_accuracy, _ = mean_confidence_interval(accuracies)
                print("Final Val_Accu = {}".format(val_accuracy))

                # Save Model
                if val_accuracy > max_accuracy:
                    # Prepare folder
                    folder_for_this_accuracy = os.path.join(output_folder, str(val_accuracy))
                    max_accuracy = val_accuracy
                    print("Models Saved with accuracy={}".format(max_accuracy))
                else:
                    folder_for_this_accuracy = os.path.join(output_folder, "Latest")

                if not os.path.exists(folder_for_this_accuracy):
                    os.mkdir(folder_for_this_accuracy)

                # Save networks
                torch.save(encoder.state_dict(), os.path.join(folder_for_this_accuracy, "encoder.pkl"))
                torch.save(rn.state_dict(), os.path.join(folder_for_this_accuracy, "rn.pkl"))
                torch.save(tcn.state_dict(), os.path.join(folder_for_this_accuracy, "tcn.pkl"))
                torch.save(ap.state_dict(), os.path.join(folder_for_this_accuracy, "ap.pkl"))

                torch.save(encoder_optim.state_dict(), os.path.join(folder_for_this_accuracy, "encoder_optim.pkl"))
                torch.save(rn_optim.state_dict(), os.path.join(folder_for_this_accuracy, "rn_optim.pkl"))
                torch.save(tcn_optim.state_dict(), os.path.join(folder_for_this_accuracy, "tcn_optim.pkl"))
                torch.save(ap_optim.state_dict(), os.path.join(folder_for_this_accuracy, "ap_optim.pkl"))

                torch.save(encoder_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "encoder_scheduler.pkl"))
                torch.save(rn_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "rn_scheduler.pkl"))
                torch.save(tcn_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "tcn_scheduler.pkl"))
                torch.save(ap_scheduler.state_dict(), os.path.join(folder_for_this_accuracy, "ap_scheduler.pkl"))
    
    print("Training Done")
    print("Final Accuracy = {}".format(max_accuracy))

# Program Starts
if __name__ == '__main__':
    main()
