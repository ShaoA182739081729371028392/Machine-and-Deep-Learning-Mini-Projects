This Repo captures the details of how I created a Modified UNet + FPN and ASPP segmentation model to identify abnormalities in kidneys. 
I modified almost everything since my last commit and was able to get top 70 Public LB as the time of this commit.

Single Fold 512x512 EffNet Unet:
- 93.3LB and 93.7CV

UPDATE: I retrained the model in TensorFLow on TPUs, and was able to achieve 29th on the private LB with a private score of 94.7 using an ensemble. 
- Ensemble: 4 Folds LinkNet, 1 Fold UNet
- Public LB: 0.934, Private LB 0.947
