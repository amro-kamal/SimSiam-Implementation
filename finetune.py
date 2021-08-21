from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import logging
from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from main import SimSiam

def predict(model, test_loader,device):
    """Measures the accuracy of a model on a data set.""" 
    # Make sure the model is in evaluation mode.
    model.to(device)
    model.eval()
    preds=[]
    labels=[]
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        # Loop over test data.
        for images, targets in tqdm(test_loader):
            # Forward pass.
            output = model(images.to(device)) #[bs x out_dim]
            preds+= (output.cpu()) #[bs x 1]
            labels+=targets
   
    return preds , labels

class LC_Dataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (list): list of input fetaures.
            labels (list): list of labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.features=features
        self.labels=labels

    def __len__(self):
        return len(self.features)
    #TODO
    def __getitem__(self, idx):
        return ( self.features[idx],self.labels[idx] )

def cifar10_loader(batch_size,finetune=False):
    train_transforms = transforms.Compose([
                                        # transforms.RandomResizedCrop(32) ,
                                        transforms.ToTensor()  ,
                                        transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                                        # transforms.RandomHorizontalFlip()
    ])

    val_transforms = transforms.Compose([
                                        # transforms.Resize(32) ,
                                        transforms.ToTensor()  ,
                                        transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])


    train_data = torchvision.datasets.CIFAR10('data/train',train=True,download=True, transform=train_transforms)
    if finetune:
      train_data, _ = torch.utils.data.random_split(train_data, [10000, 40000])

    val_data = torchvision.datasets.CIFAR10('data/val',train=False,download=True, transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def train(model, train_loader , val_loader, args,criterion,optimizer,mixup=False,alpha = 1):
    """
       Simple training loop for PyTorch model.
       args:  epochs , model_path='model.ckpt' ,load_model=False, min_val_acc_to_save=88.0
    """ 
    writer=SummaryWriter(os.path.join(args.save_path,'tensorboard',args.exp_name))

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    optimizer.zero_grad()

    # Move model to the device (CPU or GPU).
    model.train()
    model.to(device)
    
    # Exponential moving average of the loss.
    losses=[]
    val_losses = []
    train_accs=[]
    val_accs=[]
    best_val_acc=0

    print(f'----- Training on {device} -----')
    # Loop over epochs.
    for epoch in range(args.epochs):
        correct = 0
        num_examples=0
        ema_loss = 0
        # Loop over data.
        loop=tqdm(enumerate(train_loader , start =epoch*len(train_loader)),leave=False, total=len(train_loader))
        for step , (images, target) in loop:

            # Forward pass.
            output = model(images.to(device))
            loss = criterion(output, target.to(device))

            # Backward pass.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            # NOTE: It is important to call .item() on the loss before summing.
            ema_loss += (loss.item() - ema_loss) * 0.01 
            # Compute the correct classifications
            preds = output.argmax(dim=1, keepdim=True)
            correct+= preds.cpu().eq(target.view_as(preds)).sum().item()
            num_examples+= images.shape[0]
            train_acc=correct/num_examples
            #tqdm
            loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            loop.set_postfix(loss=ema_loss, acc=train_acc)
        
        #write the loss to tensorboard    
        writer.add_scalar('train loss', ema_loss, global_step=epoch)
        writer.add_scalar('train acc', train_acc, global_step=epoch)
        writer.add_scalar('train error', 1-train_acc, global_step=epoch)

        losses.append(ema_loss)
        train_accs.append(train_acc)

        val_acc, val_loss= test(model ,val_loader, device,criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        writer.add_scalar('val loss', val_loss, global_step=epoch)
        writer.add_scalar('val acc', val_acc, global_step=epoch)
        writer.add_scalar('val eror', 1-val_acc, global_step=epoch)

        if val_acc > best_val_acc and val_acc > args.min_val_acc_to_save:
            print(f'validation accuracy increased from {best_val_acc} to {val_acc}  , saving the model ....')
            #saving training ckpt
            chk_point={'model_sate_dict':model.state_dict(), 'epochs':epoch+1, 'best_val_acc':best_val_acc}
            torch.save(chk_point, os.path.join(args.save_path,args.exp_name,'model.ckpt'))
            best_val_acc=val_acc
        print('-------------------------------------------------------------')
        
        if epoch+1==100 or epoch+1==150:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/10

    return train_accs , val_accs, losses, val_losses
    
def test(model, data_loader, device,criterion):
    """Measures the accuracy of a model on a data set.""" 
    # Make sure the model is in evaluation mode.
    model.eval()
    correct = 0
    total = 0
    ema_loss = 0
    print(f'----- Model Evaluation on {device}-----')
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        
        # Loop over test data.
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
  
            # Forward pass.
            outputs = model(features)
            
            # Get the label corresponding to the highest predicted probability.
            preds = outputs.argmax(dim=1, keepdim=True) #[bs x 1]
            
            # Count number of correct predictions.
            correct += preds.eq(targets.view_as(preds)).sum().item()

            loss = criterion(outputs , targets)
            ema_loss  +=  (loss.item() - ema_loss) * 0.01 

    model.train()
    # Print test accuracy.
    percent = 100. * correct / len(data_loader.sampler)
    print(f'validation accuracy: {correct} / {len(data_loader.sampler)} ({percent:.2f}%)')
    return percent , ema_loss


    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Mixup Training')

    parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--learning-rate', default=0.1, type=float,help='base learning rate')
   
    parser.add_argument('--min-val-acc-to-save', default=30.0, type=float )

    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--mode',default='linearclassifier')
    
    parser.add_argument('--ckpt-path', default = '', type=str)
    parser.add_argument('--save-path', default = '', type=str)

    parser.add_argument('--exp-name', default = 'resnet18', type=str)
    parser.add_argument('--seed', default=123, type=int)

    args=parser.parse_args()

    simsiam = SimSiam(model_path='')
    simsiam = SimSiam.load_from_checkpoint(checkpoint_path=args.ckpt_path, strict=False)#, **args.__dict__)
    
    resnet=simsiam.resnet
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    criterion = torch.nn.CrossEntropyLoss()

    if args.mode=='finetune':
      print('finetuining on 10% of cifar10')
      cifar10_train_loader, cifar10_val_loader=cifar10_loader(batch_size=args.batch_size,finetune=True)
      resnet.fc=nn.Linear(512,10)
      optimizer = torch.optim.Adam(resnet.parameters(), lr=args.learning_rate)#, momentum=0.9, weight_decay=args.weight_decay)
      train_accs , val_accs, losses, val_losses = train(resnet, cifar10_train_loader, cifar10_val_loader, args, criterion, optimizer)

    else: #'linearclassifier'

      resnet.fc=nn.Identity()
      #get cifar10 dataset
      cifar10_train_loader, cifar10_val_loader=cifar10_loader(batch_size=args.batch_size)
      #get the 2048-dim features from the model
      print('Getting the feartures from resnet')
      train_features, train_labels = predict(resnet , cifar10_train_loader ,device)
      #print(f'train_features shape {train_features.shape}')
      LC_train_dataset = LC_Dataset(train_features, train_labels)
      cifar10_train_loader = DataLoader(LC_train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=2)
    
      val_features, val_labels=predict(resnet,cifar10_val_loader,device)
      LC_val_dataset=LC_Dataset(val_features, val_labels)
      cifar10_val_loader=DataLoader(LC_val_dataset,batch_size=args.batch_size,shuffle=True,num_workers=2)
      del resnet
      linear_classifer=nn.Sequential(nn.Linear(512,10))
      optimizer=torch.optim.Adam(linear_classifer.parameters(), lr=0.3)
      train_accs , val_accs, losses, val_losses = train(linear_classifer, cifar10_train_loader, cifar10_val_loader, args, criterion, optimizer)
