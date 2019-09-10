import numpy as np
import os
import joblib
import re
import cv2

CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", 
    "walking"]

# Dataset are divided according to the instruction at:
# http://www.nada.kth.se/cvap/actions/00sequences.txt
TRAIN_PEOPLE_ID = [11, 12, 13, 14, 15, 16, 17, 18]
DEV_PEOPLE_ID = [19, 20, 21, 23, 24, 25, 1, 4]
TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]

def make_data(data_path, raw_folder, processed_folder, sequence_name='00sequences.txt'):
    
    train = []
    dev = []
    test = []
    sequence = match_sequence(os.path.join(data_path, raw_folder, sequence_name))

    frames = []

    n_processed_files = 0

    for category in CATEGORIES:
        print("Processing category %s" % category)

        # Get all files in current category's folder.
        folder_path = os.path.join(data_path, raw_folder, category)
        filenames = os.listdir(folder_path)

        for filename in filenames:
            filepath = os.path.join(folder_path, filename)
            vid = cv2.VideoCapture(filepath)

            # Store features in current file.
            frames_current_file = []

            while vid.isOpened():
                ret, frame = vid.read()

                if not ret:
                    break

                frame =  cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
                frames_current_file.append(frame)
        
            if 'person13_handclapping_d3' in sequence[filename[:-11]]:
                continue

            if len(frames_current_file) < sequence[filename[:-11]][-1]:
                print('problem: {}, seq {}/c {}, fixed'.format(filename[:-11], sequence[filename[:-11]][-1], len(frames_current_file)))
                sequence[filename[:-11]][-1] = len(frames_current_file)



            frames.append({
                'filename': filename,
                'category': category,
                'frames': frames_current_file,
                'sequence': sequence[filename[:-11]],
            })

            n_processed_files += 1
            if n_processed_files % 100 == 0:
                    print("Done %d files. (total: 600)" % n_processed_files)

    return split_data(data_path, processed_folder, frames)

def split_data(data_path, processed_folder, all_data):

    train_id = list(map(lambda x: str(x).zfill(2), TRAIN_PEOPLE_ID))
    test_id = list(map(lambda x: str(x).zfill(2), TEST_PEOPLE_ID))
    validate_id = list(map(lambda x: str(x).zfill(2), DEV_PEOPLE_ID))

    train_data = []
    test_data = []
    validate_data = []

    def find_id(file_name):
        return re.findall(r"([0-9]{2})", file_name)[0]

    # 生成 train的
    for notes in all_data:
        notes_id = find_id(notes['filename'])

        if notes_id in train_id:
            train_data.append(notes)
        if notes_id in test_id:
            test_data.append(notes)
        if notes_id in validate_id:
            validate_data.append(notes)
    
    # Dump data to file.
    # print('Dumping train data to file...')
    # joblib.dump(all_data, open(os.path.join(data_path, processed_folder, 'train.pkl'), "wb"))
    # print('Dumping test data to file...')
    # joblib.dump(all_data, open(os.path.join(data_path, processed_folder, 'test.pkl'), "wb"))
    # print('Dumping validate data to file...')
    # joblib.dump(all_data, open(os.path.join(data_path, processed_folder, 'validate.pkl'), "wb"))
    return train_data, test_data, validate_data

def match_sequence(path):
    with open(path, 'r') as f:
        data = f.read()
        h = re.findall(r"((?:person\d\d_\w{1,}_\w{1,})|(?:\d{1,4}))", data)

    start = False
    list_ = []
    all_list = []
    for note in h:
        if 'person' in note and start == False:
            start = True
            list_.append(note)
        elif 'person' not in note and start == True:
            list_.append(note)
        elif 'person' in note and start == True:
            all_list.append(list_)
            list_ = []
            list_.append(note)
    all_list.append(list_)

    dic = {}

    for note in all_list:
        if 'person13_handclapping_d3' in note[0]:
            continue
        to_add = [1,]
        for i in range(2,len(note)-1):
            if int(note[i])+1 == int(note[i+1]) or int(note[i])-1 == int(note[i-1]):
                continue
            else:
                to_add.append(int(note[i]))
        to_add.append(int(note[-1]))
        dic[note[0]] = to_add

    return dic

if __name__ == "__main__":
    # print(os.getcwd())
    # match_sequence('./data/KTHDataset/raw/00sequences.txt')
    make_data(data_path='./data/KTHDataset', raw_folder='raw', processed_folder='processed', sequence_name='00sequences.txt')
    pass
