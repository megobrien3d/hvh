# hvh

This repository contains the code for the implementation of "half-versus-half" (hvh) 
multiclass SVM (and its variations), which is a method of multiclass SVM inspired by 
the existing methods one-versus-rest and one-versus-one. Like the other two methods,
the half-versus-half method combines a number of binary classifiers in order to 
determine a classification.

The original version of half-versus-half I devised begins by creating a number of 
different partitions of the classes into different halves. These different halves
are then separated by a binary linear classifier, and the compilation of all of 
these classifiers is then used to uniquely identify each class. The partitions are
chosen to be such that log(ceil(len(N))) planes (where N is the number of classes)
are required which is a significant reduction from the N planes required for 
one-versus-rest and the N*(N-1)/2 planes required for one-versus-one. 

The results on this original version are modest: using the HOG_01, HOG_02, and 
HOG_03 datasets from GTSRB, hvh was able to obtain the following results: 

-- Num classes used --  | -- Data Set --  | -- Classification Accuracy on Test Data --  | Num planes used
          8             |       HOG_01    |                   1.0                       |       3
          8             |       HOG_02    |                   1.0                       |       3
          8             |       HOG_03    |                   1.0                       |       3
          10            |       HOG_01    |                   0.9995                    |       4
          10            |       HOG_02    |                   0.9896                    |       4
          10            |       HOG_03    |                   0.991                     |       4
          16            |       HOG_01    |                   0.98                      |       4
          16            |       HOG_02    |                   0.9966                    |       4
          16            |       HOG_03    |                   0.9993                    |       4
          20            |       HOG_01    |                   0.9448                    |       5
          20            |       HOG_02    |                   0.986                     |       5
          20            |       HOG_03    |                   0.99                      |       5
          32            |       HOG_01    |                   0.891                     |       5
          32            |       HOG_02    |                   0.9385                    |       5
          32            |       HOG_03    |                   0.9353                    |       5
          42            |       HOG_01    |                   0.8213                    |       6
          42            |       HOG_02    |                   0.8767                    |       6
          42            |       HOG_03    |                   0.8665                    |       6
          
More tests demonstrated the high variability of the half-versus-half method on the 
particular partitioning into halves chosen. Because the goal is to save computation 
at the time of classification rather than saving computation at training time, the 
method was modified (see vary_partitions.py) to repeat the process of choosing a 
partition multiple times by shuffling the initial state of the list of classes. The
original half-versus-half fitting would then proceed normally and the particular 
partition with the highest accuracy on the test data would be selected to be the 
set of planes to comprise the half-versus-half classifier. 

The results showed that the method works well on small numbers of classes but as the
number of classes increased, the accuracy fell. This suggests that though the classes
may originally be linearly separable, when grouped together larger collections as this
halving method entails, these larger super-classes/groups are not necessarily linearly
separable anymore, so a further modification was devised that would increase the 
number of planes slightly but would hopefully increase the accuracy of the method on 
datasets with a larger number of classes. The modified version (found in the file
modify_partitions.py), rather than finding a minimal partition in the algorithmic 
way the original version of hvh (which can be examined in the functions "partition"
and "alternate" in the file hvh.py), it instead randomly splits the classes in half
and checks if these two halves are sufficiently linearly separable ("sufficiently" 
as determined by the user in the parameter MAX_ERROR). The algorithm then keeps those
planes and halvings which are good and skips those halvings which are not able to be
separated by a binary classifier very well. The algorithm continues to find such halves
until the classes all have a unique position relative to each of the different planes. 
In practice, this did not significantly increase the accuracy of half-versus-half. 

Considering that half-versus-half works well with a smaller number of classes (up to 
around 20 classes), it might be a worthwhile investigation to see if datasets with a 
very large number of classes can be split into linearly separable super-groupings with
size about ~20 classes large, and then use hvh on those super-groupings of ~20 classes. 
This would not reduce the number of planes used as much as the original version but 
still has much potential to reduce the number of computations compared to the current 
methods of one-versus-rest and one-versus-one, and may have greater accuracy than these 
original attempts at this method. 





