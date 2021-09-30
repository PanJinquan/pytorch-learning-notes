#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets ,models , transforms
import json
from torch.utils.data import Dataset, DataLoader ,random_split
from PIL import Image
from pathlib import Path
classLabels = ["desert", "mountains", "sea", "sunset", "trees" ]
print(torch.__version__)


df = pd.DataFrame({"image": sorted([ int(x.name.strip(".jpg")) for x in Path("image_scene_data/original").iterdir()])})
df.image = df.image.astype(np.str)
print(df.dtypes)
df.image = df.image.str.cat([".jpg"]*len(df))
for label in classLabels:
  df[label]=0
with open("image_scene_data/labels.json") as infile:
    s ="["
    s = s + ",".join(infile.readlines())
    s = s+"]"
    s = np.array(eval(s))
    s[s<0] = 0
    df.iloc[:,1:] = s
df.to_csv("data.csv",index=False)
print(df.head(10))
del df


# ## Visulaize the data
#

# ### Data distribution


df = pd.read_csv("data.csv")
fig1, ax1 = plt.subplots()
df.iloc[:,1:].sum(axis=0).plot.pie(autopct='%1.1f%%',shadow=True, startangle=90,ax=ax1)
ax1.axis("equal")
plt.show()


# ### Correlation between different classes

# In[ ]:


import seaborn as sns
sns.heatmap(df.iloc[:,1:].corr(), cmap="RdYlBu", vmin=-1, vmax=1)
# looks like there is no correlation between the labels


# ### Visualize images

# In[ ]:


def visualizeImage(idx):
  fd = df.iloc[idx]
  image = fd.image
  label = fd[1:].tolist()
  print(image)
  image = Image.open("image_scene_data/original/"+image)
  fig,ax = plt.subplots()
  ax.imshow(image)
  ax.grid(False)
  classes =  np.array(classLabels)[np.array(label,dtype=np.bool)]
  for i , s in enumerate(classes):
    ax.text(0 , i*20  , s , verticalalignment='top', color="white", fontsize=16, weight='bold')
  plt.show()

visualizeImage(52)


# In[ ]:


#Images in the dataset have different sizes to lets take a mean size while resizing 224*224
l= []
for i in df.image:
  with Image.open(Path("image_scene_data/original")/i) as f:
    l.append(f.size)
np.array(l).mean(axis=0),np.median(np.array(l) , axis=0)


# ## Create Data pipeline

# In[ ]:


class MyDataset(Dataset):
  def __init__(self , csv_file , img_dir , transforms=None ):

    self.df = pd.read_csv(csv_file)
    self.img_dir = img_dir
    self.transforms = transforms

  def __getitem__(self,idx):
    # d = self.df.iloc[idx.item()]
    d = self.df.iloc[idx]
    image = Image.open(self.img_dir/d.image).convert("RGB")
    label = torch.tensor(d[1:].tolist() , dtype=torch.float32)

    if self.transforms is not None:
      image = self.transforms(image)
    return image,label

  def __len__(self):
    return len(self.df)


# In[ ]:


batch_size=32
transform = transforms.Compose([transforms.Resize((224,224)) , 
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])

dataset = MyDataset("data.csv" , Path("image_scene_data/original") , transform)
valid_no = int(len(dataset)*0.12) 
trainset ,valset  = random_split( dataset , [len(dataset) -valid_no  ,valid_no])
print(f"trainset len {len(trainset)} valset len {len(valset)}")
dataloader = {"train":DataLoader(trainset , shuffle=True , batch_size=batch_size),
              "val": DataLoader(valset , shuffle=True , batch_size=batch_size)}


# ## Model Definition

# In[ ]:


model = models.resnet50(pretrained=True) # load the pretrained model
num_features = model.fc.in_features # get the no of on_features in last Linear unit
print(num_features)
## freeze the entire convolution base
for param in model.parameters():
  param.requires_grad_(False)


# In[ ]:


def create_head(num_features , number_classes ,dropout_prob=0.5 ,activation_func =nn.ReLU):
  features_lst = [num_features , num_features//2 , num_features//4]
  layers = []
  for in_f ,out_f in zip(features_lst[:-1] , features_lst[1:]):
    layers.append(nn.Linear(in_f , out_f))
    layers.append(activation_func())
    layers.append(nn.BatchNorm1d(out_f))
    if dropout_prob !=0 : layers.append(nn.Dropout(dropout_prob))
  layers.append(nn.Linear(features_lst[-1] , number_classes))
  return nn.Sequential(*layers)

top_head = create_head(num_features , len(classLabels)) # because ten classes
model.fc = top_head # replace the fully connected layer

model


# ## Optimizer and Criterion

# In[ ]:


import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
sgdr_partial = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.005 )


# ## Training

# In[ ]:


from tqdm import trange
from sklearn.metrics import precision_score,f1_score

def train(model , data_loader , criterion , optimizer ,scheduler, num_epochs=5):

  for epoch in trange(num_epochs,desc="Epochs"):
    result = []
    for phase in ['train', 'val']:
      if phase=="train":     # put the model in training mode
        model.train()
        scheduler.step()
      else:     # put the model in validation mode
        model.eval()
       
      # keep track of training and validation loss
      running_loss = 0.0
      running_corrects = 0.0  
      
      for data , target in data_loader[phase]:
        #load the data and target to respective device
        data , target = data.to(device)  , target.to(device)

        with torch.set_grad_enabled(phase=="train"):
          #feed the input
          output = model(data)
          #calculate the loss
          loss = criterion(output,target)
          preds = torch.sigmoid(output).data > 0.5
          preds = preds.to(torch.float32)
          
          if phase=="train"  :
            # backward pass: compute gradient of the loss with respect to model parameters 
            loss.backward()
            # update the model parameters
            optimizer.step()
            # zero the grad to stop it from accumulating
            optimizer.zero_grad()


        # statistics
        running_loss += loss.item() * data.size(0)
        running_corrects += f1_score(target.to("cpu").to(torch.int).numpy() ,preds.to("cpu").to(torch.int).numpy() , average="samples")  * data.size(0)
        
        
      epoch_loss = running_loss / len(data_loader[phase].dataset)
      epoch_acc = running_corrects / len(data_loader[phase].dataset)

      result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    print(result)


# In[ ]:


train(model,dataloader , criterion, optimizer,sgdr_partial,num_epochs=10)


# ## Saving & Loading model

# In[ ]:


def createCheckpoint(filename=Path("./LatestCheckpoint.pt")):
  checkpoint = {
              'epoch': 5,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              "batch_size":batch_size,
  } # save all important stuff
  torch.save(checkpoint , filename)
createCheckpoint()


# In[ ]:


# Load
'''
First Intialize the model and then just load it
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

'''

checkpoint = torch.load(Path("./LatestCheckpoint.pt"))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
batch_size = checkpoint['batch_size']

model.eval() ## or model.train()
optimizer


# ## LrFinder and One Cycle Policy
# 
# For faster convergence 

# In[ ]:


def unfreeze(model,percent=0.25):
  l = int(np.ceil(len(model._modules.keys())* percent))
  l = list(model._modules.keys())[-l:]
  print(f"unfreezing these layer {l}",)
  for name in l:
    for params in model._modules[name].parameters():
      params.requires_grad_(True)

def check_freeze(model):
  for name ,layer in model._modules.items():
    s = []
    for l in layer.parameters():
      s.append(l.requires_grad)
    print(name ,all(s))


# In[ ]:


# unfreeze 40% of the model
unfreeze(model ,0.40)
# check which layer is freezed or not
check_freeze(model)


# ### LR finder

# In[ ]:


class LinearScheduler(lr_scheduler._LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of iterations."""
    def __init__(self, optimizer, end_lr, num_iter):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearScheduler,self).__init__(optimizer)

    def get_lr(self):
        # increement one by one
        curr_iter = self.last_epoch + 1
        # get the ratio
        pct = curr_iter / self.num_iter
        # calculate lr with this formulae start + pct * (end-start)
        return [base_lr + pct * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialScheduler(lr_scheduler._LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of iterations."""

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialScheduler,self).__init__(optimizer)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        pct = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** pct for base_lr in self.base_lrs]
      
class CosineScheduler(lr_scheduler._LRScheduler):
    """Cosine increases the learning rate between two boundaries over a number of iterations."""

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(CosineScheduler,self).__init__(optimizer)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        pct = curr_iter / self.num_iter
        cos_out = np.cos(np.pi * pct) + 1
        return [self.end_lr + (base_lr - self.end_lr )/2 *cos_out for base_lr in self.base_lrs]


# In[ ]:


class LRFinder:
  
  def __init__(self, model  , optimizer , criterion ,start_lr=1e-7, device=None):
    
    self.model = model
    # Move the model to the proper device
    self.optimizer = optimizer
    self.criterion = criterion
    
    ## save the model intial dict
    self.save_file = Path("tmpfile")
    torch.save(self.model , self.save_file)    
    if device is None:
      self.device = next(model.parameters()).device
    else:
      self.device = device
    self.model.to(self.device)
    
    self.history = {"lr":[] , "losses":[]}
    for l in self.optimizer.param_groups:
      l["initial_lr"]=start_lr
    
    
  def reset(self):
    """ Resets the model to intial state """
    self.model = torch.load(self.save_file)
    self.model.train()
    self.save_file.unlink()
    print("model reset done")
    return self.model
    
  def calculateSmmothingValue(self ,beta):
    n ,mov_avg=0,0
    while True :
      n+=1
      value = yield
      mov_avg = beta*mov_avg +(1-beta)*value
      smooth = mov_avg / (1 - beta **n )
      yield smooth
    
  def lrfind(self, trainLoader,end_lr=10,num_iter=150,step_mode="exp", loss_smoothing_beta=0.99, diverge_th=5): 
        """
         Performs the lrfind test

         Arguments:
            trainLoader : The data loader
            end_lr :  The maximum lr
            num_iter : Max iteratiom
            step_mode : The anneal function by default `exp` but can be either `linear` or `cos`
            smooth_f : The loss smoothing factor, value should be between [0 , 1[
            diverge_th: The max loss value after which training should be stooped
        """
              # Reset test results
        self.history = {"lr": [], "losses": []}
        self.best_loss = None
        self.smoothner = self.calculateSmmothingValue(loss_smoothing_beta)
        
        if step_mode.lower()=="exp":
          lr_schedule = ExponentialScheduler(self.optimizer , end_lr  , num_iter,)
        elif step_mode.lower()=="cos":
          lr_schedule = CosineScheduler(self.optimizer , end_lr  , num_iter)
        elif step.mode.lower()=="linear":
          lr_schedule = LinearScheduler(self.optimizer , end_lr  , num_iter)
        else:
          raise ValueError(f"expected mode is either {exp , cos ,linear} got {step_mode}")
        
        if 0 < loss_smoothing_beta >=1:
          raise ValueError("smooth_f is outside the range [0, 1[")
        
        iterator = iter(trainLoader)
        for each_iter in range(num_iter):
          try:
            data , target = next(iterator)
          except StopIteration:
            iterator = iter(trainLoader)
            data , target = next(iterator)
         
          loss = self._train_batch(data , target)
          
          # Update the learning rate
          lr_schedule.step()
          self.history["lr"].append(lr_schedule.get_lr()[0])
          # Track the best loss and smooth it if smooth_f is specified
          if each_iter == 0:
              self.best_loss = loss
          else:
              next(self.smoothner)
              self.best_loss = self.smoothner.send(loss)
              if loss < self.best_loss:
                  self.best_loss = loss

          # Check if the loss has diverged; if it has, stop the test
          self.history["losses"].append(loss)
          if loss > diverge_th * self.best_loss:
              print("Stopping early, the loss has diverged")
              break

        print("Learning rate search finished. See the graph with {finder_name}.plot()")            
  
  def _train_batch(self,data,target):
    # set to training mode
    self.model.train()
    #load data to device
    data ,target = data.to(self.device) ,target.to(self.device)
    
    #forward pass
    self.optimizer.zero_grad()
    output = self.model(data)
    loss = self.criterion(output,target)
    
    #backward pass
    loss.backward()
    self.optimizer.step()
    return loss.item()
  
  def plot(self):
    losses = self.history["losses"]
    lr = self.history["lr"]
    plt.semilogx(lr,losses)
    plt.xlabel("Learning rate")
    plt.ylabel("Losses ")
    plt.show()


# In[ ]:


lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.lrfind(dataloader["train"], end_lr=10, step_mode="exp")


# In[ ]:


lr_finder.plot()
model= lr_finder.reset()


# In[ ]:





# ###  One Cycle Policy

# In[ ]:


class Stepper():
    "Used to \"step\" from start,end (`vals`) over `n_iter` iterations on a schedule defined by `func`"
    
    def __init__(self, val, n_iter:int, func):
        self.start,self.end = val
        self.n_iter = max(1,n_iter)
        self.func = func
        self.n = 0

    def step(self):
        "Return next value along annealed schedule."
        self.n += 1
        return self.func(self.start, self.end, self.n/self.n_iter)

    @property
    def is_done(self):
        "Return `True` if schedule completed."
        return self.n >= self.n_iter
    
# Annealing functions
def annealing_no(start, end, pct):
    "No annealing, always return `start`."
    return start
  
def annealing_linear(start, end, pct):
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start + pct * (end-start)
  
def annealing_exp(start, end, pct):
    "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start * (end/start) ** pct

def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out


# In[ ]:




class OneCyclePolicy:
  
  def __init__(self,model , optimizer , criterion ,num_iteration,num_epochs,max_lr, momentum = (0.95,0.85) , div_factor=25 , pct_start=0.4, device=None ):
    
    self.model =model
    self.optimizer = optimizer
    self.criterion = criterion
    self.num_epochs = num_epochs
    if device is None:
      self.device = next(model.parameters()).device
    else:
      self.device = device
      
    n = num_iteration * self.num_epochs
    a1 = int(n*pct_start)
    a2 = n-a1
    self.phases = ((a1 , annealing_linear) , (a2 , annealing_cos))
    min_lr = max_lr/div_factor
    self.lr_scheds = self.steps((min_lr,max_lr) , (max_lr,min_lr/1e4))
    self.mom_scheds = self.steps(momentum , momentum[::-1])
    self.idx_s = 0
    self.update_lr_mom(self.lr_scheds[0].start,self.mom_scheds[0].start)
  
  def steps(self, *steps):
      "Build anneal schedule for all of the parameters."
      return [Stepper(step, n_iter, func=func)for (step,(n_iter,func)) in zip(steps, self.phases)]

  def train(self, trainLoader , validLoader ):
    self.model.to(self.device)
    data_loader = {"train":trainLoader , "val":validLoader}
    for epoch in trange(self.num_epochs,desc="Epochs"):
      result = []
      for phase in ['train', 'val']:
        if phase=="train":     # put the model in training mode
          model.train()
        else:     # put the model in validation mode
          model.eval()

        # keep track of training and validation loss
        running_loss = 0.0
        running_corrects = 0  

        for data , target in data_loader[phase]:
          #load the data and target to respective device
          data , target = data.to(device)  , target.to(device)

          with torch.set_grad_enabled(phase=="train"):
            #feed the input
            output = self.model(data)
            #calculate the loss
            loss = self.criterion(output,target)
            preds = torch.sigmoid(output).data > 0.5
            preds = preds.to(torch.float32)

            if phase=="train"  :
              # backward pass: compute gradient of the loss with respect to model parameters 
              loss.backward()
              # update the model parameters
              self.optimizer.step()
              # zero the grad to stop it from accumulating
              self.optimizer.zero_grad()
              self.update_lr_mom(self.lr_scheds[self.idx_s].step() ,self.mom_scheds[self.idx_s].step() )

              if self.lr_scheds[self.idx_s].is_done:
                self.idx_s += 1
          
          # statistics
          running_loss += loss.item() * data.size(0)
          running_corrects += f1_score(target.to("cpu").to(torch.int).numpy() ,preds.to("cpu").to(torch.int).numpy() , average="samples")  * data.size(0)


        epoch_loss = running_loss / len(data_loader[phase].dataset)
        epoch_acc = running_corrects/ len(data_loader[phase].dataset)

        result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
      print(result)

  def update_lr_mom(self,lr=0.001,mom=0.99):
    for l in self.optimizer.param_groups:
      l["lr"]=lr
      if isinstance(self.optimizer , ( torch.optim.Adamax,torch.optim.Adam)):
          l["betas"] = ( mom, 0.999)
      elif isinstance(self.optimizer, torch.optim.SGD):
          l["momentum"] =mom


# In[ ]:


fit_one_cycle = OneCyclePolicy(model ,optimizer , criterion,num_iteration=len(dataloader["train"].dataset)//batch_size  , num_epochs =8, max_lr =1e-5 ,device=device)
fit_one_cycle.train(dataloader["train"],dataloader["val"])


# ## unfreeze 60 % architecture and retrain

# In[ ]:


# unfreeze 60% of the model
unfreeze(model ,0.60)
# check which layer is freezed or not
check_freeze(model)


# In[ ]:


lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.lrfind(dataloader["train"], end_lr=10, step_mode="exp")


# In[ ]:


lr_finder.plot()


# In[ ]:


model= lr_finder.reset()


# In[ ]:


fit_one_cycle = OneCyclePolicy(model ,optimizer , criterion,num_iteration=len(dataloader["train"].dataset)//batch_size  , num_epochs =10, max_lr =1e-5 ,device=device)
fit_one_cycle.train(dataloader["train"],dataloader["val"])


# ## unfreeze 70% model and retrain

# In[ ]:


# unfreeze 60% of the model
unfreeze(model ,0.80)
# check which layer is freezed or not
check_freeze(model)


# In[ ]:


lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.lrfind(dataloader["train"], end_lr=10, step_mode="exp")


# In[ ]:


lr_finder.plot()


# In[ ]:


model= lr_finder.reset()


# In[ ]:


fit_one_cycle = OneCyclePolicy(model ,optimizer , criterion,num_iteration=len(dataloader["train"].dataset)//batch_size  , num_epochs =10, max_lr =1e-3 ,device=device)
fit_one_cycle.train(dataloader["train"],dataloader["val"])


# ## Visualizing some end result

# In[ ]:


image , label = next(iter(dataloader["val"]))
image = image.to(device)
label = label.to(device)
output = 0
with torch.no_grad():
  output = model(image)
  output = torch.sigmoid(output)
output = output>0.2  


# In[ ]:





# In[ ]:


mean , std = torch.tensor([0.485, 0.456, 0.406]),torch.tensor([0.229, 0.224, 0.225])
def denormalize(image):
  image = image.to("cpu").clone().detach()
  image = transforms.Normalize(-mean/std,1/std)(image) #denormalize
  image = image.permute(1,2,0) 
  image = torch.clamp(image,0,1)
  return image.numpy()

def visualize(image , actual , pred):
  fig,ax = plt.subplots()
  ax.imshow(denormalize(image))
  ax.grid(False)
  classes =  np.array(classLabels)[np.array(actual,dtype=np.bool)]
  for i , s in enumerate(classes):
    ax.text(0 , i*20  , s , verticalalignment='top', color="white", fontsize=16, weight='bold')
  
  classes =  np.array(classLabels)[np.array(pred,dtype=np.bool)]
  for i , s in enumerate(classes):
    ax.text(160 , i*20  , s , verticalalignment='top', color="black", fontsize=16, weight='bold')

  plt.show()

visualize(image[1] , label[1].tolist() , output[1].tolist())


# In[ ]:


visualize(image[0] , label[0].tolist() , output[0].tolist())


# In[ ]:


visualize(image[2] , label[2].tolist() , output[2].tolist())


# In[ ]:


visualize(image[7] , label[7].tolist() , output[7].tolist())


# In[ ]:





# # Summary
# 
# ## The Final accuracy or more precisely the f1 score is 88.85%
# ## The loss is 0.1962
# 
