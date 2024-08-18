# DL-Assignment-2

This repository contains code for training a Convolutional Neural Network (CNN) image classifier using PyTorch. The CNN model is trained on the iNaturalist 12K dataset to classify images into 10 different classes of natural objects such as plants, animals, insects, etc.

## Part A: Training the CNN Model from the scratch

### Description

Part A consists of a Python script 'train_PartA.py' that trains a CNN model using PyTorch from scratch. The script allows for customization of various parameters such as the number of filters, activation function, filter organization, data augmentation, batch normalization, dropout probability, epochs, and batch size.

To train the CNN model with default settings, you can run:

```bash
python train_PartA.py --num_filters 64 --activation ReLU --filter_organization same --data_augmentation False --batch_normalization True --dropout 0.3 --epoch 10 --batch_size 64
```


## Part B: Training the CNN Model using pre - trained model

### Description

Part B consists of a Python script 'train_PartB.py' that uses pretrained model (ResNET50) using PyTorch. Further I am applying fine tuning over the pre trained model using the approach - Freeze all layers except the last layer, Freeze layers up to a certain depth, Layer-wise fine-tuning. The script allows for customization of various parameters such as the number of filters, activation function, filter organization, data augmentation, batch normalization, dropout probability, epochs, and batch size.

To train the CNN model with default settings, you can run:

```bash
python train_PartB.py --num_filters 64 --activation ReLU --filter_organization same --data_augmentation False --batch_normalization True --dropout 0.3 --epoch 10 --batch_size 64
```

## Result: 
epochs = 10 <br>
batch_size = 32 <br>
num_filters = 128 <br>
activation = 'GELU' <br>
filter_organization = 'same' <br>
data_augmentation = False<br>
batch_norm = True<br>
dropout = 0.3<br>


This is my best configuration, getting the test accuracy of 36.1% when we used the model and train it from the scratch, and getting 72.05% when we used pretrained model(ResNET50) and fine tuned it. 
