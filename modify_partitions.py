# reduce_mult.py 

from __future__ import division

import sys
NUM_CLASS = int(sys.argv[1])
DATA_SET = sys.argv[2]
TRAIN_PROP = float(sys.argv[3])
NUM_TESTS = int(sys.argv[4])
IS_TEST = False
NUM_PTS = int(sys.argv[5])
MAX_ERROR = float(sys.argv[6])

#MAX_ERROR = 0.1

#NUM_PTS = 100

print(sys.argv)

# import math
import time
import random
import math
import os
import numpy as np
from sklearn import svm

print("---------------------")
print("Imported libraries ...")

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

# German traffic signs dataset
# DATA_SET = "HOG_01"
# NUM_CLASS = 16
# Try with 16, 32, and then figure out how to make work with 41, 
# maybe by just using "41"
# DIRECTORY = '/Users/megobrien/Dropbox/NeuralNetResearch/DATA/'
DIRECTORY = '/afs/crc.nd.edu/user/m/mobrie31/Private/DATA/'

# ------------------------------------------------------------
# Read in the data
# ------------------------------------------------------------
DATA = []
LABELS = []
TEST_CLASSES = []

TRAIN_DATA_W_LABELS = []
TEST_DATA_W_LABELS = []
# TRAIN_PROP = 0.8

TRAIN_DATA = []
TRAIN_LABELS = []
TEST_DATA = []
TEST_LABELS = []

if DATA_SET == "HOG_01" or DATA_SET == "HOG_02" or DATA_SET == "HOG_03":
    # HOG has 42 classes total, select random subset
    TEST_CLASSES = random.sample(list(range(42)), NUM_CLASS) 
    # TEST_CLASSES = [9, 17, 3, 13, 22, 31, 1, 0]
    # TEST_CLASSES = [9, 17, 3, 13]

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

# -----------------------------------------------------------
# Find NUM_PTS outside and inside points for the class to do 
# planes fitting with because it will reduce computation. 
# -----------------------------------------------------------
def find_subsets_of_pts(imgs, labels, classes):
    # Find the centroids of the different classes
    centroids = []
    subset_imgs = []    
    subset_lbls = []
    for cls in classes:
        class_imgs = [imgs[i] for i in range(len(labels)) if labels[i] == cls]
        centroid = np.zeros(len(class_imgs[0]))
        for curr_img in class_imgs:
            centroid = np.add(centroid, curr_img)
        centroids.append(np.log(np.asarray(centroid) / len(class_imgs)))

        # Find the outlying points for the classes (farthest points from the 
        # centroid, etc.)
        dist_to_centroid = []
        for curr_img in class_imgs:
            dist_to_centroid.append(dist_betw_pts(curr_img, centroid))
        
        # Take all these distances and take those points which have the highest
        # distance to the centroid    
        # Changed this to be the smallest distance to the centroid since it SEEMS
        # like this data is associated more to the fit on all the training data
        # which does best. But I also don't know this for sure.  
        dist_w_pt = zip(dist_to_centroid, class_imgs)
        dist_w_pt.sort(key = lambda x: x[0], reverse=False)

        subset = dist_w_pt[:NUM_PTS] #+ dist_w_pt[-NUM_PTS:]        
        subset_imgs += [pt for _, pt in subset] 
        subset_lbls += [cls]*len(subset)

    return subset_imgs, subset_lbls
  
def dist_pt_to_plane(pt, plane, bias):
    temp = np.dot(pt, plane)
    temp = np.add(temp, bias)
    magn = math.sqrt(np.dot(plane, plane))
    return temp / magn

def dist_betw_pts(pt1, pt2):
    assert len(pt1) == len(pt2)
    return math.sqrt(sum([ (pt1[i] - pt2[i])**2 for i in range(len(pt1)) ]) )

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

    assert goal_len == len(my_list)

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

def remove_none(tuples_list):
    return_list = []
    for xs, ys in tuples_list:
        new_x = [x for x in xs if x is not None]
        new_y = [y for y in ys if y is not None]

        return_list.append((new_x, new_y))
    
    return return_list

# Train the SVMs to obtain all the necessary planes
def hvh_make_planes(data, labels, class_list):
    #random.shuffle(class_list)
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
    
        # TODO
        # CHECK IF THIS PLANE IS "GOOD" (i.e. if this superclass is linearly separable)
        # IF NOT, CHANGE THE PARTITION


    if IS_TEST:
        print('----------------------------')
    cls_to_hash_dict = make_hash_dict(partitions, class_list)  

    return lsh_matrices, lsh_biases, cls_to_hash_dict, partitions 


# Instead of doing the systematic partition, do a random 
# halving each time, and check if those two superclasses
# are fairly linearly separable and if they are, keep 
# this separation (otherwise, try a new halving) 
# and then shuffle the classes again and keep adding 
# "good" planes (those which separate linearly separable
# superclasses) until we have a unique classification for
# each class.    
def make_planes_sep(imgs, labels, classes):
    lsh_matrices = []
    lsh_biases = []
    cls_to_hash_dict = {}
    partitions = []    
    
    classes_no_nones = [cls for cls in classes if cls]

    # Keep adding a plane until the hash is unique
    while not hash_unique(partitions, classes_no_nones):
        lin_sep = False
        iters = 0
        while not lin_sep:
            my_classes = list(classes)
            random.shuffle(my_classes)
            cut_off = int(len(my_classes)/2)
            super_cls_1 = my_classes[:cut_off]
            super_cls_2 = my_classes[cut_off:]
            temp_partition = (super_cls_1, super_cls_2)

            temp_labels = [int(labels[i] in super_cls_1) for i in range(len(labels))] # True is 1, Flase is 0
            clf = svm.SVC(kernel='linear', C = 1.0)
            clf.fit(imgs, temp_labels)
            
            temp_acc = sum(clf.predict(imgs) == temp_labels) / len(labels)

            if (temp_acc >= 1 - MAX_ERROR):
                #print("Keeping current plane.")
                lin_sep = True
                lsh_matrices.append(clf.coef_[0])
                lsh_biases.append(clf.intercept_[0])
                partitions.append(temp_partition)
                
            elif iters >= 30:
                print("Took too many tries. Using plane anyways.")
                lin_sep = True
                lsh_matrices.append(clf.coef_[0])
                lsh_biases.append(clf.intercept_[0])
                partitions.append(temp_partition)
 
    # After the fact, find the cls_to_hash_dict 
    cls_to_hsh_dict = find_cls_to_hsh_dict(partitions, classes)

    return lsh_matrices, lsh_biases, cls_to_hsh_dict, partitions

def make_planes_fixed_partition(imgs, labels, classes, partitions):
    lsh_matrices = []
    lsh_biases = []
    cls_to_hash_dict = {}
   
    #print("Partitions: " + str(partitions))
     
    for super_cls_1, super_cls_2 in partitions:
        temp_labels = [int(labels[i] in super_cls_1) for i in range(len(labels))] # True is 1, Flase is 0
        clf = svm.SVC(kernel='linear', C = 1.0)
        clf.fit(imgs, temp_labels)
        
        lsh_matrices.append(clf.coef_[0])
        lsh_biases.append(clf.intercept_[0])
        
    cls_to_hsh_dict = find_cls_to_hsh_dict(partitions, classes)
    return lsh_matrices, lsh_biases, cls_to_hsh_dict, partitions

def hash_unique(partitions, classes):
    hash_set = set()

    for cls in classes:
        curr_hash = []
        for cls_1, _ in partitions:
            curr_hash.append(str(int(cls in cls_1)))
        curr_hash = ''.join(curr_hash)
        
        if curr_hash in hash_set:
            return False
        else:
            hash_set.add(curr_hash)
    
    return True
 
def find_cls_to_hsh_dict(partitions, classes):
    cls_to_hsh_dict = {} 

    for cls in classes:
        curr_hash = []
        for cls_1, _ in partitions:
            curr_hash.append(int(cls in cls_1))
        #curr_hash = ''.join(curr_hash)
        
        cls_to_hsh_dict[cls] = curr_hash

    return cls_to_hsh_dict

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


all_perc_correct_train_2 = []
all_perc_correct_test_2 = []


subset_imgs, subset_lbls = find_subsets_of_pts(DATA, LABELS, TEST_CLASSES)
assert len(subset_imgs) == len(subset_lbls)

goal_len = np.power(2, np.int(np.ceil(np.log2(len(TEST_CLASSES)))))
num_to_add = goal_len - len(TEST_CLASSES)
cls_list = list(TEST_CLASSES) #+ [None]*num_to_add

for i in range(NUM_TESTS): 
    random.shuffle(cls_list)

    #lsh_matrix, lsh_biases, cls_to_hash_dict, partitions = hvh_make_planes(subset_imgs, subset_lbls, cls_list) 
    
    lsh_matrix, lsh_biases, cls_to_hash_dict, partitions = make_planes_sep(subset_imgs, subset_lbls, cls_list)
 
    # XXX Technically only trained with a small proportion of the "train" data 
    perc_correct_train, perc_incorrect_train, perc_cannot_classify_train = acc_hvh(lsh_matrix, lsh_biases, cls_to_hash_dict, list(TRAIN_DATA_W_LABELS))
    
    perc_correct_test, perc_incorrect_test, perc_cannot_classify_test = acc_hvh(lsh_matrix, lsh_biases, cls_to_hash_dict, list(TEST_DATA_W_LABELS))
    
    # Tuple containing the order of the cls_list which forms the partition that gives the current perc_correct_train 
    all_perc_correct_train.append( (partitions, perc_correct_train, len(lsh_biases)))
    all_perc_incorrect_train.append(perc_incorrect_train)
    all_perc_cannot_classify_train.append(perc_cannot_classify_train)

    all_perc_correct_test.append( (partitions, perc_correct_test, len(lsh_biases)))
    all_perc_incorrect_test.append(perc_incorrect_test)
    all_perc_cannot_classify_test.append(perc_cannot_classify_test)
    
print("***************************")

print("STATS ON SUBSET FITTING")

print("all_perc_correct_train: " + str([(perc, num_planes) for _, perc, num_planes in all_perc_correct_train]))
#print("all_perc_incorrect_train: " + str(all_perc_incorrect_train))
#print("all_perc_cannot_classify_train: " + str(all_perc_cannot_classify_train))


print("all_perc_correct_test: " + str([(perc, num_planes) for _, perc, num_planes in all_perc_correct_test]))
#print("all_perc_incorrect_test: " + str(all_perc_incorrect_test))
#print("all_perc_cannot_classify_test: " + str(all_perc_cannot_classify_test))

print("***************************")

optimal_partition, max_train_perc_correct, optimal_acc_num_planes = max(all_perc_correct_train, key = lambda x: x[1])

print("max train percent correct: " + str( max(all_perc_correct_train, key = lambda x: x[1])[1] ) )
print("max test percent correct: " + str( max(all_perc_correct_test, key = lambda x: x[1])[1] ) )
print("min number of planes: " + str( min(all_perc_correct_train, key = lambda x: x[2])[2] ) )


print("----------------------------------")
print("ALL TRAINING DATA FITTING WITH OPTIMAL ACCURACY PARTITION")
print("----------------------------------")

# Take optimal partition and now do a fit on ALL the training data with this particular partition.
#lsh_matrix, lsh_biases, cls_to_hash_dict, partitions = hvh_make_planes(DATA, LABELS, optimal_partition) 


def my_argmax_acc(my_list):
    argmax_candidates = []
    curr_max = my_list[0][1]    

    for ind in range(len(my_list)): 
        if my_list[ind][1] > curr_max:
            argmax_candidates = [ind]
            curr_max = my_list[ind][1] 
        elif my_list[ind][1] == curr_max:
            argmax_candidates.append(ind)

    if len(argmax_candidates) == 1:
        return argmax_candidates[0]
        print("argmax: " + str(argmax_candidates[0]))
    # Resolving the ties by choosing the one with the fewest number
    # of planes used
    elif len(argmax_candidates) > 1:
        print("max val: " + str(curr_max))
        print("argmax_candidates: " + str(argmax_candidates))

        min_planes = my_list[argmax_candidates[0]][2]
        min_arg = 0
        for arg in argmax_candidates:   
            if my_list[arg][2] < min_planes:
                min_planes = my_list[arg][2]
                min_arg = arg
        return min_arg      
    else: 
        raise ValueError("argmax_candidates is empty???")


#arg_opt_acc = np.argmax( [x[1] for x in all_perc_correct_train] )
#lsh_matrix, lsh_biases, cls_to_hash_dict, partitions = make_planes_fixed_partition(DATA, LABELS, cls_list, optimal_partition)
arg_opt_acc = my_argmax_acc(all_perc_correct_train)

print("arg_opt_acc: " + str(arg_opt_acc))

max_planes_partition, max_planes_acc, optimal_acc_num_planes = all_perc_correct_train[arg_opt_acc] 
lsh_matrix, lsh_biases, cls_to_hash_dict, partitions = make_planes_fixed_partition(DATA, LABELS, cls_list, max_planes_partition) 

perc_correct_train, perc_incorrect_train, perc_cannot_classify_train = acc_hvh(lsh_matrix, lsh_biases, cls_to_hash_dict, list(TRAIN_DATA_W_LABELS))
perc_correct_test, perc_incorrect_test, perc_cannot_classify_test = acc_hvh(lsh_matrix, lsh_biases, cls_to_hash_dict, list(TEST_DATA_W_LABELS))

print("Resub accuracy: " + str(perc_correct_train))
print("Test accuracy: " + str(perc_correct_test))
print("Number of planes used: " + str(optimal_acc_num_planes))
print("Partition id: " + str(arg_opt_acc))

print("----------------------------------")
print("ALL TRAINING DATA FITTING WITH MINIMAL NUMBER OF PLANES")
print("----------------------------------")

# Take optimal partition and now do a fit on ALL the training data with this particular partition.
#lsh_matrix, lsh_biases, cls_to_hash_dict, partitions = hvh_make_planes(DATA, LABELS, optimal_partition) 


def my_argmin_num_planes(my_list):
    argmin_candidates = []
    curr_min = my_list[0][2]    

    for ind in range(len(my_list)):
        if my_list[ind][2] < curr_min:
            argmin_candidates = [ind] 
            curr_min = my_list[ind][2]
        elif my_list[ind][2] == curr_min:
            argmin_candidates.append(ind)

    if len(argmin_candidates) == 1:
        return argmin_candidates[0]
    # Resolving the ties by choosing the one with the fewest number
    # of planes used
    elif len(argmin_candidates) > 1:
        max_acc = my_list[argmin_candidates[0]][1]
        max_arg = 0
        for arg in argmin_candidates:   
            if my_list[arg][1] > max_acc:
                max_acc = my_list[arg][1]
                max_arg = arg
        return max_arg 
    else: 
        raise ValueError("argmin_candidates is empty???")

arg_min_planes = my_argmin_num_planes(all_perc_correct_train)
min_planes_partition, min_planes_acc, min_planes_num_planes = all_perc_correct_train[arg_min_planes] 
#arg_min_planes = np.argmin( [x[2] for x in all_perc_correct_train] )
#min_planes_partition, min_planes_acc, min_planes_num_planes = min(all_perc_correct_train, key = lambda x: x[2])

if arg_min_planes != arg_opt_acc: 

    print("Using arg: " + str(arg_min_planes))
    print("Corresponding to training accuracy of: " + str(all_perc_correct_train[arg_min_planes][1]))

    lsh_matrix, lsh_biases, cls_to_hash_dict, partitions = make_planes_fixed_partition(DATA, LABELS, cls_list, min_planes_partition)

    perc_correct_train, perc_incorrect_train, perc_cannot_classify_train = acc_hvh(lsh_matrix, lsh_biases, cls_to_hash_dict, list(TRAIN_DATA_W_LABELS))
    perc_correct_test, perc_incorrect_test, perc_cannot_classify_test = acc_hvh(lsh_matrix, lsh_biases, cls_to_hash_dict, list(TEST_DATA_W_LABELS))

    print("Resub accuracy: " + str(perc_correct_train))
    print("Test accuracy: " + str(perc_correct_test))
    print("Number of planes used: " + str(min_planes_num_planes))
    print("Partition id: " + str(arg_min_planes))
else:
    print("The two partitions are the same.")

print("----------------------------------")
print("----------------------------------")

# Compute resubstitution and test accuracy on 1v1

clf = svm.LinearSVC()
clf.fit(TRAIN_DATA, TRAIN_LABELS)

Result=clf.predict(TRAIN_DATA)
acc_on_train_1v1 = sum(Result == TRAIN_LABELS) / len(TRAIN_LABELS)

Results2 = clf.predict(TEST_DATA)
acc_on_test_1v1 = sum(Results2 == TEST_LABELS) / len(TEST_LABELS)

print("1v1 accuracy on training data: " + str(acc_on_train_1v1))
print("1v1 accuracy on test data: " + str(acc_on_test_1v1))


print("***************************")
print("***************************")
print("***************************")
print("***************************")
print("***************************")
print("***************************")
print("***************************")
print("***************************")
print("***************************")
print("***************************")
