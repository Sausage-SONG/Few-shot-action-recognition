import os

# Constant (Settings)
TCN_OUT_CHANNEL = 64                        # Num of channels of output of TCN
RELATION_DIM = 32                           # Dim of one layer of relation net
CLASS_NUM = 3                               # <X>-way  | Num of classes
SAMPLE_NUM = 5                              # <Y>-shot | Num of supports per class
<<<<<<< HEAD
QUERY_NUM = 5                               # Num of instances for query per class
=======
QUERY_NUM = 3                               # Num of instances for query per class
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e
TRAIN_EPISODE = 30000                       # Num of training episode
TRAIN_FREQUENCY = 3                         # One data loading every <X> episode
VALIDATION_EPISODE = 50                     # Num of validation episode
VALIDATION_FREQUENCY = 200                  # After each <X> training episodes, do validation once
<<<<<<< HEAD
LEARNING_RATE = 0.0005                      # Initial learning rate
=======
LEARNING_RATE = 0.0005                       # Initial learning rate
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e
FRAME_NUM = 10                              # Num of frames per clip
CLIP_NUM = 5                                # Num of clips per window
WINDOW_NUM = 3                              # Num of processing window per video
INST_NUM = 15                               # Num of videos selected in each class
<<<<<<< HEAD
GPU = ""                                    # Index of GPU to be used
EXP_NAME = ""                               # Name of this experiment

# Dataset
##################################################################################################################
DATASET = ""             # "haa" or "mit" or "finegym"

TRAIN_SPLIT = ""                
VAL_SPLIT = ""
TEST_SPLIT = ""                    

HAA_DATA_FOLDERS = ["",
                    "",
                    ""]

MIT_DATA_FOLDERS = ["",
                    ""]

FINEGYM_DATA_FOLDER = ""
=======
GPU = "4,5,6,7"                               # Index of GPU to be used
EXP_NAME = "CTC+MSE_Full_Finegym_NewRN_3W5S"    # Name of this experiment
LOG_NAME = EXP_NAME+'.txt'

# Dataset
##################################################################################################################
DATASET = "finegym"             # "haa" or "mit" or "finegym"

TRAIN_SPLIT = "/data/ssongad/finegym/train.txt"
VAL_SPLIT = "/data/ssongad/finegym/train.txt"
TEST_SPLIT = "/data/ssongad/finegym/test.txt"

HAA_DATA_FOLDERS = ["/data/ssongad/haa/frame/train",
                    "/data/ssongad/haa/frame/val"]

MIT_DATA_FOLDERS = ["/data/ssongad/mit2/mot_normalized_frame/train",
                    "/data/ssongad/mit2/mot_normalized_frame/val"]

FINEGYM_DATA_FOLDER = "/data/ssongad/finegym/frame"
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e
##################################################################################################################

# Saved Models & Optimizers & Schedulers
##################################################################################################################
<<<<<<< HEAD
MAX_ACCURACY = 0.0            # Accuracy of the loaded model
                                             # Leave 0 if N/A

CHECKPOINT = ""             # Path of a folder, if you put everything in this folder with their DEFAULT names
=======
MAX_ACCURACY = 0.304            # Accuracy of the loaded model
                                             # Leave 0 if N/A

CHECKPOINT = "/data/ssongad/codes/new_rn/models/CTC+MSE_Full_Finegym_NewRN_3W5S/0.304"             # Path of a folder, if you put everything in this folder with their DEFAULT names
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e
                            # If you have such a path, paths below are not necessary then
                            # Leave a blank string if N/A

ENCODER_MODEL = ""          # 
RN_MODEL = ""               # Path of saved models
TCN_MODEL = ""              # Leave a blank string if N/A            
<<<<<<< HEAD
=======
RN0_MODEL = ""              # 
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e
AP_MODEL = ""               # 

ENCODER_OPTIM = ""          # 
RN_OPTIM = ""               # Path of saved optimizers                                      
TCN_OPTIM = ""              # Leave a blank string if N/A
<<<<<<< HEAD
=======
RN0_OPTIM = ""              # 
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e
AP_OPTIM = ""               # 

ENCODER_SCHEDULER = ""      # 
RN_SCHEDULER = ""           # Path of saved schedulers
TCN_SCHEDULER = ""          # Leave a blank string if N/A
<<<<<<< HEAD
AP_SCHEDULER = ""           # 
=======
RN0_SCHEDULER = ""          # 
AP_SCHEDULER = ""          # 
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e


# If CHECKPOINT is given, then use files under CHECKPOINT first
# Only use the specific path of a file when it's missing under CHECKPOINT 
if os.path.exists(CHECKPOINT):
    results = []
<<<<<<< HEAD
    default_names = ("encoder.pkl", "rn.pkl", "tcn.pkl", "encoder_optim.pkl", "rn_optim.pkl", "tcn_optim.pkl",
                    "encoder_scheduler.pkl", "rn_scheduler.pkl", "tcn_scheduler.pkl", "ap.pkl", "ap_optim.pkl", 
                    "ap_scheduler.pkl")
=======
    default_names = ("encoder.pkl", "rn.pkl", "tcn.pkl", "rn0.pkl", "encoder_optim.pkl", "rn_optim.pkl", "tcn_optim.pkl",
                     "rn0_optim.pkl", "encoder_scheduler.pkl", "rn_scheduler.pkl", "tcn_scheduler.pkl", "rn0_scheduler.pkl",
                     "ap.pkl", "ap_optim.pkl", "ap_scheduler.pkl")
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e
    for default_name in default_names:
        tmp = os.path.join(CHECKPOINT, default_name)
        results.append(tmp if os.path.exists(tmp) else "")
    
    ENCODER_MODEL = results[0] if results[0] != "" else ENCODER_MODEL
    RN_MODEL = results[1] if results[1] != "" else RN_MODEL
    TCN_MODEL = results[2] if results[2] != "" else TCN_MODEL
<<<<<<< HEAD
    AP_MODEL = results[9] if results[9] != "" else AP_MODEL

    ENCODER_OPTIM = results[3] if results[3] != "" else ENCODER_OPTIM
    RN_OPTIM = results[4] if results[4] != "" else RN_OPTIM
    TCN_OPTIM = results[5] if results[5] != "" else TCN_OPTIM
    AP_OPTIM = results[10] if results[10] != "" else AP_OPTIM

    ENCODER_SCHEDULER = results[6] if results[6] != "" else ENCODER_SCHEDULER
    RN_SCHEDULER = results[7] if results[7] != "" else RN_SCHEDULER
    TCN_SCHEDULER = results[8] if results[8] != "" else TCN_SCHEDULER
    AP_SCHEDULER = results[11] if results[11] != "" else AP_SCHEDULER
=======
    RN0_MODEL = results[3] if results[3] != "" else RN0_MODEL
    AP_MODEL = results[12] if results[12] != "" else AP_MODEL

    ENCODER_OPTIM = results[4] if results[4] != "" else ENCODER_OPTIM
    RN_OPTIM = results[5] if results[5] != "" else RN_OPTIM
    TCN_OPTIM = results[6] if results[6] != "" else TCN_OPTIM
    RN0_OPTIM = results[7] if results[7] != "" else RN0_OPTIM
    AP_OPTIM = results[13] if results[13] != "" else AP_OPTIM

    ENCODER_SCHEDULER = results[8] if results[8] != "" else ENCODER_SCHEDULER
    RN_SCHEDULER = results[9] if results[9] != "" else RN_SCHEDULER
    TCN_SCHEDULER = results[10] if results[10] != "" else TCN_SCHEDULER
    RN0_SCHEDULER = results[11] if results[11] != "" else RN0_SCHEDULER
    AP_SCHEDULER = results[14] if results[14] != "" else AP_SCHEDULER
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e
##################################################################################################################


# Dataset Split
##################################################################################################################
def read_split(file_path):
    result = []
    if os.path.exists(file_path):
        file = open(file_path, "r")
        lines = file.readlines()
        file.close()

        for line in lines:
            result.append(line.rstrip())
            
    return result

TRAIN_SPLIT = read_split(TRAIN_SPLIT)
VAL_SPLIT = read_split(VAL_SPLIT)
TEST_SPLIT = read_split(TEST_SPLIT)

if DATASET == "haa":
    DATA_FOLDERS = HAA_DATA_FOLDERS
elif DATASET == "finegym":
    DATA_FOLDERS = FINEGYM_DATA_FOLDER
elif DATASET == "mit":
    DATA_FOLDERS = MIT_DATA_FOLDERS
##################################################################################################################
<<<<<<< HEAD
INFO_DICT = ["", ""]
=======
INFO_DICT = ["/data/ssongad/finegym/gym288_train_element_v1.1.txt", "/data/ssongad/finegym/gym288_val_element.txt"]
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e
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
<<<<<<< HEAD
INFO_DICT = d
=======
INFO_DICT = d
>>>>>>> 9600d47f2f5e1e3e0932d09541f121528ccd979e
