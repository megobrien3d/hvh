# vary_partition.py 

from __future__ import division

import time

import sys
NUM_CLASS = int(sys.argv[1])
DATA_SET = sys.argv[2]
TRAIN_PROP = float(sys.argv[3])
NUM_TESTS = int(sys.argv[4])
IS_TEST = False

print(sys.argv)

# import math
import time
import random
import os
import numpy as np
from sklearn import svm

print("---------------------")
print("Imported libraries ...")

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DIRECTORY = '/afs/crc.nd.edu/user/m/mobrie31/Private/DATA/'

# ------------------------------------------------------------
# Read in the data
# ------------------------------------------------------------
DATA = []
LABELS = []
TEST_CLASSES = []

TRAIN_DATA_W_LABELS = []
TEST_DATA_W_LABELS = []

TRAIN_DATA = []
TRAIN_LABELS = []
TEST_DATA = []
TEST_LABELS = []

if DATA_SET == "HOG_01" or DATA_SET == "HOG_02" or DATA_SET == "HOG_03":
    # HOG has 42 classes total, select random subset
    TEST_CLASSES = random.sample(list(range(42)), NUM_CLASS) 

    for cls in TEST_CLASSES:
        class_name = str(cls).zfill(5) + '/'
        curr_dir = DIRECTORY + 'GTSRB_Final_Training_HOG/GTSRB/Final_Training/HOG/' + DATA_SET + '/' + class_name
        for filename in os.listdir(curr_dir):
            filename = curr_dir + filename
            temp = open(filename).read().splitlines()
            temp = [float(x) for x in temp]
            DATA.append(temp)
            LABELS.append(cls)
    
        ALL_DATA = zip(DATA, LABELS)
        random.shuffle(ALL_DATA)
        TRAIN_DATA_W_LABELS = ALL_DATA[:int(TRAIN_PROP * len(ALL_DATA))]
        TEST_DATA_W_LABELS = ALL_DATA[int(TRAIN_PROP * len(ALL_DATA)):]
        
        TRAIN_DATA, TRAIN_LABELS = [x for x, y in TRAIN_DATA_W_LABELS], [y for x, y in TRAIN_DATA_W_LABELS]
        TEST_DATA, TEST_LABELS = [x for x, y in TEST_DATA_W_LABELS], [y for x, y in TEST_DATA_W_LABELS]          

print("Data loaded ...")
print("---------------------")
# ------------------------------------------------------------
# Find a partition into halves which uniquely identifies each
# class element
# ------------------------------------------------------------

def alternate(jump_by, my_list):
    list1 = []
    list2 = []

    first = True
    curr_index = 0

    while curr_index < len(my_list):
        if first:
            first = False
            list1 += my_list[curr_index: curr_index + jump_by]
        else:
            first = True
            list2 += my_list[curr_index: curr_index + jump_by]
        curr_index += jump_by

    return (list1, list2)


def partition(my_list):
    # Convert the list into a list of strings (if not already)
    # my_list = [str(x) for x in my_list]
    goal_len = np.power(2, np.int(np.ceil(np.log2(len(my_list)))))

    # Extend length of my_list to be next power of 2
    my_list += [None]*(goal_len - len(my_list))
    random.shuffle(my_list)
    
    # Create the partitions
    partitions = []
    jump_by = np.int(np.ceil(len(my_list)/2))
    for _ in range(np.int(np.ceil((len(my_list))))):
        if jump_by > 0:
            partitions.append(alternate(jump_by, my_list))
            jump_by = int(jump_by/2)

    return partitions


# ------------------------------------------------------------
# Create the lsh hash based off of a partitioning into 
# different sets of uniquely identifying halves.
# ------------------------------------------------------------
def make_hash_dict(partition, class_list):
    hash_dict = {}

    for cls in class_list: 
        temp_hash = []
        for target, _ in partition:
            if cls in target: 
                temp_hash.append(1)
            else:
                temp_hash.append(0)
        hash_dict[cls] = temp_hash

    return hash_dict

def remove_none(tuples_list):
    return_list = []
    for xs, ys in tuples_list:
        new_x = [x for x in xs if x is not None]
        new_y = [y for y in ys if y is not None]

        return_list.append((new_x, new_y))
    
    return return_list

# Train the SVMs to obtain all the necessary planes
def hvh_make_planes(data, labels, class_list):
    random.shuffle(class_list)
    partitions = partition(class_list)
    partitions = remove_none(partitions)    

    lsh_matrices = []
    lsh_biases = []
    
    if IS_TEST:
        print('----------------------------')
    
    for targets_1, targets_2 in partitions:
        xs = data
        ys = []
        for vec, lbl in zip(data, labels):
            if lbl in targets_1: 
                ys.append(1)
            else: 
                ys.append(0)
        
        clf = svm.SVC(kernel='linear', C = 1.0)
        clf.fit(xs,ys)
        lsh_matrices.append(clf.coef_[0])
        lsh_biases.append(clf.intercept_[0])        
 
        if IS_TEST:
            percent_done = (100*len(lsh_biases))/len(partitions)    
            print(str(percent_done) + "% of clf fitting done...")

    if IS_TEST:
        print('----------------------------')
    cls_to_hash_dict = make_hash_dict(partitions, class_list)  

    return lsh_matrices, lsh_biases, cls_to_hash_dict, partitions 


# ------------------------------------------------------------
# Test out hvh
# ------------------------------------------------------------

def classify(vec, cls_to_hash_dict):
    for cls, hsh in cls_to_hash_dict.items():
        if np.all(np.equal(vec, hsh)):
            return cls
    
    # If none of the hashes are identical to the hash vec, 
    # admit that we don't know what it is
    return None 

def acc_hvh(lsh_matrices, lsh_biases, cls_to_hash_dict, test_data): 
    num_right = 0
    num_incorrect = 0
    num_cannot_classify = 0    

    for vec, lbl in test_data:
        temp = np.matmul(lsh_matrices, vec)
        temp = np.add(temp, lsh_biases)
        temp = np.sign(temp)
        temp = np.clip(temp, 0, 1) 
        
        guess = classify(temp, cls_to_hash_dict)

        if guess == lbl:
            num_right += 1
            
        elif guess == None: 
            num_cannot_classify += 1

        else: 
            num_incorrect += 1
 
    return num_right/len(test_data), num_incorrect/len(test_data), num_cannot_classify/len(test_data) 


# Calculate resubstition and test accuracy on hvh 

print("test classes: " + str(TEST_CLASSES))

all_perc_correct_train = []
all_perc_incorrect_train = []
all_perc_cannot_classify_train = []

all_perc_correct_test = []
all_perc_incorrect_test = []
all_perc_cannot_classify_test = []


for _ in range(NUM_TESTS):  
    lsh_matrix, lsh_biases, cls_to_hash_dict, partitions = hvh_make_planes(DATA, LABELS, TEST_CLASSES) 
    
    perc_correct_train, perc_incorrect_train, perc_cannot_classify_train = acc_hvh(lsh_matrix, lsh_biases, cls_to_hash_dict, list(TRAIN_DATA_W_LABELS))
    
    perc_correct_test, perc_incorrect_test, perc_cannot_classify_test = acc_hvh(lsh_matrix, lsh_biases, cls_to_hash_dict, list(TEST_DATA_W_LABELS))

    all_perc_correct_train.append(perc_correct_train)
    all_perc_incorrect_train.append(perc_incorrect_train)
    all_perc_cannot_classify_train.append(perc_cannot_classify_train)

    all_perc_correct_test.append(perc_correct_test)
    all_perc_incorrect_test.append(perc_incorrect_test)
    all_perc_cannot_classify_test.append(perc_cannot_classify_test)

print("***************************")

print("all_perc_correct_train: " + str(all_perc_correct_train))
print("all_perc_incorrect_train: " + str(all_perc_incorrect_train))
print("all_perc_cannot_classify_train: " + str(all_perc_cannot_classify_train))


print("all_perc_correct_test: " + str(all_perc_correct_test))
print("all_perc_incorrect_test: " + str(all_perc_incorrect_test))
print("all_perc_cannot_classify_test: " + str(all_perc_cannot_classify_test))

print("***************************")

print("max train percent correct: " + str(max(all_perc_correct_train)))
print("max test percent correct: " + str(max(all_perc_correct_test)))

print("***************************")

# Compute resubstitution and test accuracy on 1v1

clf = svm.LinearSVC()
clf.fit(TRAIN_DATA, TRAIN_LABELS)

Result=clf.predict(TRAIN_DATA)
acc_on_train_1v1 = sum(Result == TRAIN_LABELS) / len(TRAIN_LABELS)

Results2 = clf.predict(TEST_DATA)
acc_on_test_1v1 = sum(Results2 == TEST_LABELS)/ len(TEST_LABELS)

print("1v1 accuracy on training data: " + str(acc_on_train_1v1))
print("1v1 accuracy on test data: " + str(acc_on_test_1v1))


print("***************************")
print("***************************")
