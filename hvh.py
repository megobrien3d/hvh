# hvh.py 

# IDEA: The approximately log number of planes idea.
# (Where divide the classes into different halves, so that among
# all the sets of halves, no one class is in all the same halves
# as another class)

from __future__ import division

import time

import sys
NUM_CLASS = int(sys.argv[1])
DATA_SET = sys.argv[2]
TRAIN_PROP = float(sys.argv[3])

print(sys.argv)

# import math
import time
import random
import os
import numpy as np
from sklearn import svm

print("---------------------")
print("Imported libraries...")

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

print("Data loaded...")


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

    my_list += [None]*(goal_len - len(my_list))
    random.shuffle(my_list)

    # Append None to my_list until reaching a list whose length is a power of 2
    #while len(my_list) < goal_len:
        #rand_index = random.randint(0, len(my_list) - 1)
        #my_list.insert(rand_index, None)

    # Make log2(len(my_list)) partitions of my_list
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

# Train the SVMs to obtain all the necessary planes
def hvh_make_planes(data, labels, class_list):
    partitions = partition(class_list)
    lsh_matrices = []
    lsh_biases = []

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

        temp_vec_is_set = False
        temp_vec = [0]*len(DATA[0])  # number of dimensions of the space
        for j in range(0, len(DATA[0])):
            # if never enters this if statement, that is an error
            if clf.coef_[0][j] != 0 and not temp_vec_is_set:
                temp_vec[j] = -1*clf.intercept_[0] / clf.coef_[0][j]
                temp_vec_is_set = True
                break

        if (not temp_vec_is_set):
            print("BAD. Temp_vec not set, which doesn't make sense.")


        temp_mul = np.matmul(np.asarray(temp_vec), lsh_matrices[-1])
        lsh_biases.append(temp_mul)
        
        percent_done = (100*len(lsh_biases))/len(partitions)    
        print(str(percent_done) + "% of clf fitting done...")

    print('----------------------------')
    cls_to_hash_dict = make_hash_dict(partitions, class_list)  

    return lsh_matrices, lsh_biases, cls_to_hash_dict 


# ------------------------------------------------------------
# Test out hvh
# ------------------------------------------------------------

def classify(vec, cls_to_hash_dict):
    most_similar = None 
    max_dig_same = 0    

    for cls, hsh in cls_to_hash_dict.items():
        temp_sum = np.sum(np.equal(vec, hsh))        
        if temp_sum > max_dig_same:
            max_dig_same = temp_sum
            most_similar = cls        

        if np.all(np.equal(vec, hsh)):
            return cls
    
    # Or should I return None if we don't definitively know?    
    if most_similar == None: 
        return random.sample(list(cls_to_hash_dict))

    return most_similar 

         

# Ask Andras if how I should make guesses on vectors that don't 
# hash to a bucket that has a class in it. (Only a problem for 
# when the number of classes isn't a power of 2.)

def acc_hvh(lsh_matrices, lsh_biases, cls_to_hash_dict, test_data): 
    num_right = 0

    for vec, lbl in test_data:
        temp = np.matmul(lsh_matrices, vec)
        temp = np.subtract(temp, lsh_biases)
        temp = np.sign(temp)
        temp = np.clip(temp, 0, 1) 
        
        if classify(temp, cls_to_hash_dict) == lbl:
            num_right += 1
 
        #if np.all(np.equal(cls_to_hash_dict[lbl], temp)):
         #   num_right += 1
        #else if shares_most(temp, cls_to_hash_dict) == lbl:
         #   num_right += 1
        

    print("num_right: " + str(num_right))
    print("len(test_data): " + str(len(test_data)))    

    return num_right / len(test_data)


# Calculate resubstition and test accuracy on hvh 
start_time = time.time()
lsh_matrices, lsh_biases, cls_to_hash_dict = hvh_make_planes(DATA, LABELS, TEST_CLASSES) 
hvh_train_time = time.time() - start_time


start_time = time.time()

hvh_acc_on_train = acc_hvh(lsh_matrices, lsh_biases, cls_to_hash_dict, list(TRAIN_DATA_W_LABELS))
hvh_acc_on_test = acc_hvh(lsh_matrices, lsh_biases, cls_to_hash_dict, list(TEST_DATA_W_LABELS))

hvh_classify_time = time.time() - start_time

print("hvh accuracy on train data: " + str(hvh_acc_on_train))
print("hvh accuracy on test data: " + str(hvh_acc_on_test))
print("hvh train time: " + str(hvh_train_time))
print("hvh classify time: " + str(hvh_classify_time))

# Compute resubstitution and test accuracy on 1v1
start_time = time.time()

clf = svm.LinearSVC()
clf.fit(TRAIN_DATA, TRAIN_LABELS)

train_time_1v1 = time.time() - start_time
start_time = time.time()

Result=clf.predict(TRAIN_DATA)
acc_on_train_1v1 = sum(Result == TRAIN_LABELS) / len(TRAIN_LABELS)

Results2 = clf.predict(TEST_DATA)
acc_on_test_1v1 = sum(Results2 == TEST_LABELS)/ len(TEST_LABELS)

classify_time_1v1 = time.time() - start_time

print("1v1 accuracy on train data: " + str(acc_on_train_1v1))
print("1v1 acuracy on test data: " + str(acc_on_test_1v1))  
print("1v1 train time: " + str(train_time_1v1))
print("1v1 classify time: " + str(classify_time_1v1))

print("xxxxxxxxxxx"+str(hvh_acc_on_train)+","+str(hvh_acc_on_test)+","+str(hvh_train_time)+","+str(hvh_classify_time)+","+str(acc_on_train_1v1)+","+str(acc_on_test_1v1)+","+str(train_time_1v1)+","+str(classify_time_1v1))
