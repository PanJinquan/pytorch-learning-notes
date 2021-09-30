## pytorch Classify Scene Images (Multi-Instance Multi-Label problem)

The objective of this study is to develop a deep learning model that 
will identify the natural scenes from images. This type of problem comes
under multi label image classification where an instance can be 
classified into multiple classes among the predefined classes. 

## Dataset

The Complete dataset description can be found on http://lamda.nju.edu.cn/data_MIMLimage.ashx

The processed data is their in the repo  [Image_data.zip](./image_scene_data.zip)

## Data Description

The image data set consists of 2,000 natural scene images, where a set of labels is artificially assigned to each image. The number of images belonging to more than one class (e.g. *sea+sunset*). On average, each image is associated with 1.24 class labels.



The `labels.json` file contains the labels in the form of list [1 -1 -1 1 -1], it means that the i-th  image belongs to the 1st and 4th classes but do not belong to the 2nd,  3rd and 5th classes. The following is the order of classes **desert, mountains, sea, sunset, trees**

## Table of Content 

1. Download Data

2. Structure the data

3. Visulaize the data
	1. Data distribution
	2. Correlation between different classes
	3. Visualize images
4. Create Data pipeline
5. Model Definition (RESNET50)
6. Optimizer(Adam) and Criterion (nn.BCEWithLogitsLoss)
7. Training
8. Saving & Loading model
9. Model Finetuning
	1. LrFinder and One Cycle Policy
	2. unfreeze 60 % architecture and retrain
	3. unfreeze 70% model and retrain

10 . Visualizing some end result

## Metric Used

1. Precision Score 
2. F1 score

Note:- Refer sklearn doc for deeper understanding of the metric

# Training

As this is multi label image classification, the loss function was  binary crossentropy logit and activation function used was sigmoid at the  output layer. so after training there is one probabilistic threshold method which  find out the best threshold value for each label seperately and based on the threshold value(0.5)

```python
preds = torch.sigmoid(output).data > 0.5
preds = preds.to(torch.float32)
```

## Final Result

F1 score| Loss
----------|-----------
88.85%|0.1962
