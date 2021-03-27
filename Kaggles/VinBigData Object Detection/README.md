This repository is where I will store the 2 stage training process to classify abnormalities in lung scans.

Training Process:
1) train a pruned EfficientNetb3 on classifying whether or not an image has an abnormality(Loss: 0.11, Accuracy: 96%)
2) Using pretrained weights from 1, use pruned EfficientNet as backbone for completely custom YOLOv3 Model.

Unfortunately, I didn't have enough time to train the model fully before the competition ended, but my code for training and inference is still listed here.
