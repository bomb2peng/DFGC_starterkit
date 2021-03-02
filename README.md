## Introcuction ##  
This the the starter-kit for the [DFGC-2021 competition](https://competitions.codalab.org/competitions/29548).
Please apply for CelebDF-v2 dataset from this [site](https://github.com/yuezunli/celeb-deepfakeforensics).

## List Files ##
* **submit_image_list.txt**: For the creation track, this is the names of specified 1000 image to be submitted. e.g. "id0_id1_0000_00060" 
stands for the *60*th frame of the target video *id0_0000.mp4* with its face swapped to *id1*. Submitted images must be 
named exactly as specified in this file, and the format can be '.png' or '.jpg'. The 1000 images should be packed into a single 
.zip file with no extra subdirectories. e.g.:  
*submission.zip*  
--id0_id1_0000_00000.png  
--id0_id1_0000_00060.png  
...  
--id58_id57_0008_00120.png

* **train-list.txt**: For the detection track, this is the names of training videos. No extra training data is allowed 
except the data in this set or obtained from this set.

* **test-list.txt**: This is the testing video set used for evaluation. *submit_image_list.txt* images are extracted from 
this set. Do NOT use this set for training detection models or model validation.

## Coming Soon... ##
* The sample submission for the detection track (code and model).  
* The training codes of a baseline Xception model for detection track.