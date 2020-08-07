import scipy.stats
import numpy as np
import time
import ctc
from torch.nn.init import kaiming_normal_

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

def time_tick(name, t0=None):
    if t0 is None:
        return time.time()
    t1 = time.time()
    time_used = round(t1 - t0, 2)
    content = "{} used {}s".format(name, time_used)
    
    return content, t1

def ctc_predict(input):
    results = []
    for i in range(len(input)):
        result, _ = ctc.decode(input[i])
        results.append(result)
    return results

def weights_init(m):
    kaiming_normal_(m.weight)