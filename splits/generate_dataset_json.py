import json
import os

def read_split(file_path):
    result = []
    if os.path.exists(file_path):
        file = open(file_path, "r")
        lines = file.readlines()
        file.close()

        for line in lines:
            result.append(line.rstrip())
            
    return result

def read_finegym_info(finegym_info):
    d = dict()
    for path in finegym_info:
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

    return d

def generate_finegym(folder="<INPUT_FINEGYM_DIR_HERE>", train_split_path="./finegym_train.txt",  
                     val_split_path="./finegym_train.txt", test_split_path="./finegym_test.txt",
                     finegym_info=["./gym288_train_element_v1.1.txt", "./gym288_val_element.txt"],
                     output_name="example_finegym.json"):
    result = dict()

    result["name"] = "finegym"
    result["folder"] = folder
    result["splits"] = [read_split(train_split_path), 
                        read_split(val_split_path), 
                        read_split(test_split_path)]

    result["finegym_info"] = read_finegym_info(finegym_info)

    with open(output_name, "w") as file:
        file.write(json.dumps(result))

def generate_standard(folders=["<INPUT_FOLDERS_HERE>"], train_split_path="<INPUT_SPLIT_PATH_HERE>",
                        val_split_path="<INPUT_SPLIT_PATH_HERE>", test_split_path="<INPUT_SPLIT_PATH_HERE>",
                        name="<INPUT_DATASET_NAME_HERE>", output_name="example_standard.json"):
    result = dict()

    result["name"] = name
    result["folders"] = folders
    result["splits"] = [read_split(train_split_path), 
                        read_split(val_split_path), 
                        read_split(test_split_path)]
    
    with open(output_name, "w") as file:
        file.write(json.dumps(result))

generate_finegym()
generate_standard()