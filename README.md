# SimSiam-Implementation
A pytorch Lightning implementation for the SimSiam  [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)

The backbone encoder is resnet18. I used the cifar10 variant of resnet18 from this [repo](https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py) which gives an extremely better results for cifar10 than torchvision resnet18 model. 

For the Knn clasifier I used [this code](https://github.com/IgorSusmelj/barlowtwins/blob/main/utils.py).  
The features dimension is 1024. The hidden dimension of the prediction MLP is 512. The output dimension on the backbone model is 512.

**Requirements:**

```
torch==1.9
pytorch-lightning==1.3.2
lightning-bolts==0.3.4
```

### Install the requirements :
```
!pip install -r req.txt
```

### To train the model:
```
!python main.py --batch-size=512 \
                --epochs=3\
                --save-dir='/simsiam'
```

## KNN classifier acccuracy after just 180 epoch is 81% (in the paper they trained the model for 800 epochs! to get 91.8%)
<img width="600" alt="Screen Shot 2021-07-25 at 00 14 44" src="https://user-images.githubusercontent.com/37993690/126883827-44a66a2e-7867-4e88-9499-83451b9f0174.png">

## Training loss
<img width="600" alt="Screen Shot 2021-07-25 at 00 15 56" src="https://user-images.githubusercontent.com/37993690/126883872-ea9d605d-2dc0-44d5-8a83-259b4702e168.png">

## Learning rate for SGD (10 epochs warm-up + consine)
<img width="600" alt="Screen Shot 2021-07-25 at 00 15 35" src="https://user-images.githubusercontent.com/37993690/126883842-90d9b3c8-4ec5-47b8-aebf-86bad534605c.png
                                                               
                                                               
#TODO
 1-Training for 800 epoch.
 2-Training a linear classifier on top of the model.
 3-Fine tuning the backbone model on 10% of the labeled data .                                                             
                                                               

