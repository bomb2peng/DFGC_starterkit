## Updates ##  
* 4-10: added a new "landms_68" field in the sample_meta.json file. We are now providing the 68 facial landmarks apart from bounding box and 5-landmark information. Landmarks are arranged in a column of "x1 y1 x2 y2 ... x68 y68". Note fake images submitted by different teams share exactly the same set of landmark labelling with the baseline fake images, which is an approximation and may have minor errors. So use this information with caution.

## Introcuction ##  
This the the starter-kit for the [DFGC-2021 competition](https://competitions.codalab.org/competitions/29548).
Please apply for CelebDF-v2 dataset from this [site](https://github.com/yuezunli/celeb-deepfakeforensics).

## List Files ##
* **submit_image_list.txt**: For the creation track, this is the names of specified 1000 image to be submitted. e.g. "id0_id1_0000_00060" 
stands for the *60*th frame of the target video *id0_0000.mp4* with its face swapped to *id1*. Submitted images must be 
named exactly as specified in this file, and the format can be '.png' or '.jpg'. Note the face-swapped image should have
exactly the same image size (width and height) as the original target frame.  
The 1000 created images should be packed into a single .zip file with no extra subdirectories. e.g.:  
**submission_swap.zip  
--id0_id1_0000_00000.png  
--id0_id1_0000_00060.png  
...  
--id58_id57_0008_00120.png**

* **train-list.txt**: For the detection track, this is the names of training videos. No extra training data is allowed 
except the data in this set or obtained from this set. The label 0 is real, and 1 is fake.

* **test-list.txt**: This is the testing video set used for evaluation. *submit_image_list.txt* images are extracted from 
this set. Do NOT use this set for training detection models or model validation.

## Example Training Code for a Baseline Detection Model ##
We provide an example for training a baseline Xception model using the CelebDF-v2 training set. Please see *Det_model_training/* .
Main requirements for this example code are Pytorch and facenet_pytorch.

The processing and training steps are as follows (remember to change to your specific data directories in the codes):  
1. cd to *Det_model_training/*  
2. Run `python preprocess/crop_face.py` to process training set to cropped faces.  
3. Run `python preprocess/make_csv.py` to produce train-val split files.
4. Run `python train_xception_baseline.py` to train the Xception model.

## Example Submission for the Detection Track ##
The submission for the detection track is a zip file with at least a single python program file named *model.py* . Other 
depended files may also be included as you need. The submission format is:  
**submission_det.zip  
--model.py  
--any_others**  
Note there must NOT be any redundant directories above *model.py* .

An example detection submission is in *submission_Det/* (after unzip). It is the baseline Xception model trained above. 
The requirement is that *model.py* has a class named *Model* and it has a *\_\_init\_\_* method and a *run* method. 
The *\_\_init\_\_* method should take no extra input argument, and it is responsible for model definement and initialization 
(model loading). The *run* method should take two input argument *input_dir* and *json_file*, where the first one is the 
input testing image directory (e.g. *sample_imgs/*) and the second one is a json file (e.g. *sample_meta.json*) with 
the face bounding box and five landmarks information 
of each testing image (detected by a MTCNN detector). You may or may not use this information in your inference code, 
but this input parameter MUST be kept. The output of the *run* method should be a list of testing image names and 
a list of predictions in exact correspondence. The prediction is the probability of testing image being fake,  
in the range [0, 1], where 0 is real and 1 is fake. You can also decide your own *batchsize* in the inference code according to your deep 
network size if you use deep learning methods, and this may speed up your submission. For more details, please read the 
example code. To run this baseline model, first download the *weights.ckpt* from [BaiduYun](https://pan.baidu.com/s/1GTGQ5qYad99JrdltQM6UKg) 
with extraction code *b759* or from [GoogleDrive](https://drive.google.com/file/d/1ieyYi2Vyd7d_QrbV6YNQeui5OvalWPEj/view?usp=sharing),
 and put it in *submission_Det/* .

**Before making a submission to the competition site, please first test it locally using the *evaluate_Detection.py*.** 
The competition server uses the *nicedif/dfgc:v2* Docker image from [here](https://hub.docker.com/layers/nicedif/dfgc/v2/images/sha256-1fcbe55a19a24ec31495ed713bb00f276ec95c8cae3cbf6964cc2bb079c87a33?context=explore).
Specific python packages in this docker are listed in *docker_packages.txt*. 
You can install this docker image to test your submissions locally as in the same environment with our evaluation server. 
The actual evaluation will be running on an Alibaba Cloud Linux server instance with a Nvidia T4 GPU (16GB GPU mem) and 
4 CPU cores and 15GB memories.

In case your inference code needs extra python packages that are not installed in our docker image, please include them in your submission zip. 
An example detection submission is in *submission_Det2* that uses efficientnet_pytorch and albumentations (it depends on imgaug) as extra dependencies 
(model weights file is omitted). 
Note how we add python paths and import these packages in the *model.py*. Again, it is highly recomended to first test your submissions locally using 
our evaluation docker image and test your package imports.

## Coming Soon... ##
* ~~The sample submission for the detection track (code and model).~~ 
* ~~The training codes of a baseline Xception model for detection track.~~
