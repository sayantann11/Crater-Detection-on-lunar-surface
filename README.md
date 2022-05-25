# Crater-Detection-on-lunar-surface

# Abstract -YOLO v5 is a family of compoundscaled object detection models trained on the 

COCO dataset, and includes simple 
functionality for Test Time Augmentation 
(TTA), model ensembling, hyperparameter 
evolution, and export to ONNX, CoreML and 
TFLite. YOLO an acronym for 'You only look 
once', is an object detection algorithm that 
divides images into a grid system. Each cell in 
the grid is responsible for detecting objects 
within itself. YOLO is one of the most famous 
object detection algorithms due to its speed and 
accuracy. For this Challenging experiment we 
have used YOLO V5 for the circular object 
(moon surface) detection using anchor boxes.

# Introduction
Yolo architecture is more like FCNN (fully 
convolutional neural network) and passes the 
image (nxn) once through the FCNN and output is 
(mxm) prediction. This the architecture is splitting 
the input image in mxm grid and for each grid 
generation 2 bounding boxes and class 
probabilities for those bounding boxes. Note that 
bounding box is more likely to be larger than the 
grid itself. Compared to other region proposal 
classification networks (fast RCNN) which 
perform detection on various region proposals and 
thus end up performing prediction multiple times 
for various regions in a image. We reframe object 
detection as a single regression problem, straight 
from image pixels to bounding box coordinates 
and class probabilities.
A single convolutional network simultaneously 
predicts multiple bounding boxes and class 
probabilities for those boxes. YOLO trains on full 
images and directly optimizes detection 
performance. This unified model has several 
benefits over traditional methods of object 

# Model Backbone 
is mainly used to extract 
important features from the given input 
image. In YOLO v5 the CSP — Cross 
Stage Partial Networks are used as a 
backbone to extract rich in informative 
features from an input image.
# Model Neck 
is mainly used to generate 
feature pyramids. Feature pyramids help 
models to generalized well on object 
scaling. It helps to identify the same object 
with different sizes and scales.Feature 
pyramids are very useful and help models 
to perform well on unseen data. There are 
other models that use different types of 
feature pyramid techniques like FPN, 
BiFPN, PANet, etc.In YOLO v5 PANet is 
used for as neck to get feature pyramids. 
Understanding Feature Pyramid Networks 
for object detection (FPN) The model 
# Head 
is mainly used to perform the final 
detection part. It applied anchor boxes on 
features and generates final output vectors 
with class probabilities, objectness scores, 
and bounding boxes. In YOLO v5 model
head is the same as the previous YOLO V3 
and V4 versions.
# Dataset 
The Circular Ellipse dataset has 29 images of the 
moon’s surface without annotations –
The images have then been split to 20 as train and 
the remaining 9 as validation for testing purpose.
As an import phase of object detection, we have 
manually assigned the annotations using make
sense ai and further generated the object
coordinates for training our model.
The associated coordinates of the objects in the 
first image. Four coordinates as there are 4 objects 
in the above image .
The overall data structure which includes the train
validation split, labels which has the coordinates 
of the bounding box for both train and validation 
as described in the earlier images.
# Methodology and Process
We Start with our manual annotation process 
using makesense.ai where we annotate each and
every image’s objects and export the annotations 
in YOLO format.
For this experiment the YOLO V5 predefined / 
pretrained model (available in Yolo’s Github’s 
website ) is extracted and initialized.
After the execution of the previous cell , the 
runtime creates a yolov5 library as shown in the 
figure below (google collab) which is nothing but 
the prerequisites for running our YOLO model .
As we have already Split the dataset into train and
validation, we are specifying the path of train and 
validation located in our collab directory. The 
executable text which is being shown in the below 
figure is inside the yolov5 directory. We are 
modifying the number of classes and the name of 
the class according to our requirements which in 
this case is only 1 class (circular class)
labelled/named as ‘circle’. 
The final step is training our model using YOLO 
V5 for a total of 150 epochs without specifying 
any call-backs or arguments. And parallelly
checking the validation using our validation 
images which was defined in the beginning. The 
training of the images happens in a batches.
# Observations and Results
To evaluate object detection models like R-CNN 
and YOLO, the mean average precision (mAP) is 
used. The mAP compares the ground-truth 
bounding box to the detected box and returns a 
score. The higher the score, the more accurate the 
model is in its detections. So for our model the 
mAP value is 0.555 which is 55%. Along with the
loss recorded was 0.171 which is really low for a 
object detection model. The below image further 
shows all the other metrics relevant.
The generated annotations of the images –
Batch of 9 
Batch of 16 
For all the above images the red bounding box 
depicts the detected circular object and does it 
with high precision. 
# Conclusion 
Actual images with annotations –
Predicted objects using YOLO V5 –
As the predicted boxes are more of less 
overlapping the annotated boxes , The IOU is high 
which says the model is well trained for detecting 
the desired objects.
Statistics of the Model –
a) Confusion matrix -
b) Recall vs Confidence graph 
c) Precision vs Confidence graph 
d) Precision vs Recall graph 
e) F1 vs Confidence graph
The statistics shows that with every increasing
epoch the model trains better and gives a higher 
overall precision making the YOLO the best in the 
business out of all the available object detection 
models.
# References 
1. https://pytorch.org/hub/ultralytics_yolov5/#:~:t
ext=YOLOv5%20%F0%9F%9A%80%20is%2
0a%20family,Model
2. https://github.com/ultralytics/yolov5
3. https://towardsdatascience.com/the-practicalguide-for-object-detection-with-yolov5-
algorithm-74c04aac4843
4. https://medium.com/axinc-ai/yolov5-thelatest-model-for-object-detectionb13320ec516b
5. https://www.makesense.ai
