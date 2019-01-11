# kaggle-human-protein-atlas-image-classification

Kaggle 2018 @ Human Protein Atlas Image Classification

35th Place Algorithm and Model.

'*' : Working methods between options.

### Models

- vgg16
- resnet50, resnet101, ...
- densenet121, densenet169 *
- inception v3, inception v4 *
- se152
- polynet
- NASNet, PNASNet

### Implementations

- Data Loader for External Datas and Merger
- Basic data augmentations
  - Rotation, Flip *
  - Channel drops
- 16 Test-Time Augmentation
- 5-Folds Cross Validation
- Simple Threshold Search Algorithm
- Ensembles
  - Test-Time Augmentation Averaging *
  - Majority Voting *
  - Fully-Connected Neural Network
    - logits -> output
    - logits + features -> output
  - XGBoost
- Loss
  - Soft F1 Loss *
  - Binary Cross Entropy *
  - Focal Loss
  - MultiLabelMarginLoss

### Train

Training with yaml configuratin files and changed few parameter.

```
$ python main.py -c conf/densenet.yaml
$ python main.py -c conf/densenet.yaml --lr 0.0001
```