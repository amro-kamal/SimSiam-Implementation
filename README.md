# SimSiam-Implementation
A pytorch Lightning implementation for SimSiam  [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)

The backbone encoder is resnet18. I used the cifar10 variant of resnet18 from this [repo](https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/cifar10_models/resnet.py) which gives an extremely better results for cifar10 than torchvision resnet18 model. 

For the Knn clasifier I used [this code](https://github.com/IgorSusmelj/barlowtwins/blob/main/utils.py). The features dimension is 1024 (The paper used features dimension of 2048 ). The hidden dimension of the prediction MLP is 512. The output dimension of the backbone model (resnet18 average pooling layer) is 512. 

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
                --epochs=800\
                --save-dir='./simsiam'        #path to save the model and tensorboard logs 
```

**Results after 800 epochs on cifar10**
| Model         | Method    | KNN acc | linear classifier|
| ------------- | ----------| --------|------------------|
| Resnet18      | SimSiam   | 88.7    | 90.1             |


### KNN classifier acccuracy  after 800 epochs of training the accuracy is 88.7% (in the paper they trained the model for 800 epochs to get 91.8%)
<img width="600" alt="Screen Shot 2021-08-21 at 15 57 19" src="https://user-images.githubusercontent.com/37993690/130327712-05e04210-16f3-427e-80ad-aa5daa999683.png">

### KNN classifier acccuracy after just 180 epoch is 81% 
<img width="600" alt="Screen Shot 2021-07-25 at 00 14 44" src="https://user-images.githubusercontent.com/37993690/126883827-44a66a2e-7867-4e88-9499-83451b9f0174.png">


### Training loss
<img width="600" alt="Screen Shot 2021-07-25 at 00 15 56" src="https://user-images.githubusercontent.com/37993690/126883872-ea9d605d-2dc0-44d5-8a83-259b4702e168.png">

### Learning rate for SGD (10 epochs warm-up + consine)
<img width="600" alt="Screen Shot 2021-07-25 at 00 15 35" src="https://user-images.githubusercontent.com/37993690/126884424-ceba149b-699e-43ee-831c-e548ee02550d.png">

 ## Ablation Study:
 I did two experiments to test the importance of two parts of the model:
 1. The stop gradient
 2. The Prediction MLP
 
The results are the same for the two experiments. If we remove either the stop gradient or the prediction MLP, the model will fall in a mode collapse giving us a trivial solution. Although the trivial solution gives a very low loss,  the accuracy stops at around 30%. 
I trained the model for 50 epochs 3 times, once without stop gradient, once without the prediction MLP, and the last time was with both of them (the baseline model). 
 
 red: **with both stop gradient and prediction MLP**
 
 orange: **without stop gradient** 
 
 blue: **without prediction MLP**
 
 
 <img width="600" alt="Screen Shot 2021-07-26 at 14 37 53" src="https://user-images.githubusercontent.com/37993690/127009033-9f65d483-a30e-4d8b-86a2-97976e7515e4.png">
 
 We can see that the when removing the stop gradient or the prediction MLP, the loss converges after just 1 epoch, at the same time the accuracy still low which means that we are facing a trivial solutions. We can summarize that with this similarity loss that simsiam uses, both the stop gradient and the prediction MLP are important ingredients of the model to avoid collapsing into trivial solution.
 
<img width="600" alt="Screen Shot 2021-07-26 at 14 37 42" src="https://user-images.githubusercontent.com/37993690/127009061-97d3cc71-1ef3-4b87-982b-b83ffbca7717.png">

 
                                                               
### TODO
 1. Training a linear classifier on top of the backbone model.
 2. Fine tuning the backbone model on 10% of the labeled data .                                                             
                                                               

