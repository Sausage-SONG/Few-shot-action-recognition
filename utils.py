import torch
import numpy as np
import os


def print_stage(name, length=65, character="-"):
    Llength = int((length - len(name)) / 2)
    Rlength = length - Llength - len(name)
    print("\n" + character*Llength + name + character*Rlength)

def print_debug(content, character="*"):
    length = len(content) + 2
    print(character*length)
    print("{} DEBUG {}".format(character, character), end='')
    print(character*(length - 9))
    print(character*length)
    print("{}{}{}".format(character, content, character))
    print(character*length)

import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def ndarray_equal(arr1, arr2):
    return len(arr1) == len(arr2) and np.count_nonzero((arr1 == arr2) == True) == len(arr1)

def compute_score(prediction, truth):
    # zeros = np.count_nonzero(prediction == 0)
    # truths = np.count_nonzero(prediction == truth)
    return 1 if len(prediction) == 1 and prediction[0] == truth else 0

# def compute_score(prediction, truth):
#     return np.count_nonzero(prediction == truth) / len(prediction)

def write_log(content, end="\n"):
    file = open("log.txt", "a")
    file.write(content+end)
    file.close()

def write_error(content, end="\n"):
    file = open("error.txt", "a")
    file.write(content+end)
    file.close()

import time
def time_tick(name, t0=None):
    if t0 is None:
        return time.time()
    t1 = time.time()
    time_used = round(t1 - t0, 2)
    content = "{} used {}s".format(name, time_used)
    
    return content, t1

from ctc.ctc_decode import decode as ctc_decode
def ctc_predict(input):
    results = []
    for i in range(len(input)):
        result, _ = ctc_decode(input[i])
        results.append(result)
    return results

from torch.nn.init import kaiming_normal_
def weights_init(m):
    try:
        kaiming_normal_(m.weight)
    except Exception:
        return

def read_split(file_path):
    result = []
    if os.path.exists(file_path):
        file = open(file_path, "r")
        lines = file.readlines()
        file.close()

        for line in lines:
            result.append(line.rstrip())
            
    return result

def ctc_length_mask(input):
    N, L = int(input.shape[0]), int(input.shape[1])
    dtype = input.dtype

    # Starting with length L (original length), and all mask true
    lengths = input.new_full((N, ), fill_value=L, dtype=torch.int, requires_grad=False)
    mask = input.new_full((N, L), fill_value=True, dtype=torch.bool, requires_grad=False)

    # Case: Length 1
    if L == 1:
        return lengths, mask
    
    # Case: Length >= 2
    for i in range(1, L):
        # If one same to last one, length--, mask->False
        lengths[input[:, i] == input[:, i-1]] -= 1
        mask[input[:, i] == input[:, i-1], i] = False
    
    return lengths, mask

from ctc.ctc_loss import ctcLabelingProb as ctc_prob
from string import ascii_letters as letters
def ctc_probability(prob, truth):
    C = int(prob.shape[1])
    C -= 1
    assert C <= 27

    classes = [letters[i] for i in range(C)]
    classes = "".join(classes)

    gt = truth - 1
    gt = [letters[i] for i in gt]

    blank = prob[:, 0]
    blank = blank.unsqueeze(1)
    non_blank = prob[:, 1:]
    mat = torch.cat((non_blank, blank), 1)

    return ctc_prob(mat, gt, classes)

def ctc_alignment_predict(probs, targets, target_lengths, sample_num):
    N, W, CL = int(probs.shape[0]), int(probs.shape[1]), int(probs.shape[2])
    CL -= 1
    S = int(target_lengths.shape[0])

    result = probs.new_full((N, ), fill_value=0, dtype=torch.int, requires_grad=False)
    
    for n in range(N):
        prediction_probs = target_lengths.new_full((S, ), fill_value=0, dtype=torch.double, requires_grad=False)
        
        start = 0
        for s in range(S):
            length = int(target_lengths[s])
            prediction_probs[s] = ctc_probability(probs[n], targets[start: start+length])
            # file = open("test.log", "a")
            # file.write("prob from "+str(probs[n])+" to "+str(targets[start: start+length])+"\n")
            # file.close()
            start += length
        
        # print("\nprediction_probs", prediction_probs, "\n")
        _, result[n] = torch.max(prediction_probs, 0)
    
    result = [int(result[i]//sample_num)+1 for i in range(len(result))]

    return result
