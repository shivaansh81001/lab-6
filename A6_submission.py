import numpy as np
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        """
        define the layers in your CNN
        """
        #refrence = https://lekhuyen.medium.com/an-overview-of-vgg16-and-nin-models-96e4bf398484
        #conv1
        self.conv1 = nn.Conv2d(3,64,kernel_size=(3,3),stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,64,kernel_size=(3,3),stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #conv2
        self.conv3 = nn.Conv2d(64,128,kernel_size=(3,3),stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128,kernel_size=(3,3),stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        #conv3
        self.conv5 = nn.Conv2d(128,256,kernel_size=(3,3),stride=1,padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256,256,kernel_size=(3,3),stride=1,padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256,256,kernel_size=(3,3),stride=1,padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        #conv4
        self.conv8 = nn.Conv2d(256,512,kernel_size=(3,3),stride=1,padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512,512,kernel_size=(3,3),stride=1,padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512,512,kernel_size=(3,3),stride=1,padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        #conv5
        self.conv11 = nn.Conv2d(512,512,kernel_size=(3,3),stride=1,padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512,512,kernel_size=(3,3),stride=1,padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512,512,kernel_size=(3,3),stride=1,padding=1)
        self.bn13 = nn.BatchNorm2d(512)

        #fully conncted 1
        self.fc1 = nn.Linear(512,512)
        self.bnfc1 = nn.BatchNorm1d(512)

        #fully conncted 2
        self.fc2 = nn.Linear(512,512)
        self.bnfc2 = nn.BatchNorm1d(512)

        #fully conncted 2
        self.fc3 = nn.Linear(512,10)

        self.dropout = nn.Dropout(0.6)

        """add code here"""

    def init_weights(self):
        """
        optionally initialize weights
        """
        """add code here"""
        #https://github.com/Xiaoming-Yu/RAGAN/blob/master/models/model.py#L50
        #https://stackoverflow.com/questions/61848635/how-to-decide-which-mode-to-use-for-kaiming-normal-initialization
        #and had to look up at GPT for understanding the implementation
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):    #weights for conv layer
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer,nn.Linear):  #Linear / fc layers weights init
                nn.init.xavier_normal_(layer.weight)
            elif isinstance(layer,(nn.BatchNorm2d, nn.BatchNorm1d)): #this is for normalization layers- (fc and conv) - normal distribution
                nn.init.constant_(layer.weight,1)
                nn.init.constant_(layer.bias,0)



    def forward(self, x):
        """
        Pass the input images through your CNN to produce the class probabilities

        :param x: batch_size x 3 x 28 x 28 tensor containing the input images
        :return: batch_size x 11 tensor containing the class probabilities for each image
        """
        out = None
        """add code here"""
        x=nn.functional.relu(self.bn1(self.conv1(x)))
        x=nn.functional.relu(self.bn2(self.conv2(x)))
        x=nn.functional.max_pool2d(x,kernel_size=(2,2),stride=2)
        #print(x.shape)

        x=nn.functional.relu(self.bn3(self.conv3(x)))
        x=nn.functional.relu(self.bn4(self.conv4(x)))
        x=nn.functional.max_pool2d(x,kernel_size=(2,2),stride=2)
        #print(x.shape)

        x=nn.functional.relu(self.bn5(self.conv5(x)))
        x=nn.functional.relu(self.bn6(self.conv6(x)))
        x=nn.functional.relu(self.bn7(self.conv7(x)))
        x=nn.functional.max_pool2d(x,kernel_size=(2,2),stride=2)
        #print(x.shape)
        
        x=nn.functional.relu(self.bn8(self.conv8(x)))
        x=nn.functional.relu(self.bn9(self.conv9(x)))
        x=nn.functional.relu(self.bn10(self.conv10(x)))
        x=nn.functional.max_pool2d(x,kernel_size=(2,2),stride=2)
        #print(x.shape)

        x=nn.functional.relu(self.bn11(self.conv11(x)))
        x=nn.functional.relu(self.bn12(self.conv12(x)))
        x=nn.functional.relu(self.bn13(self.conv13(x)))
        x=nn.functional.max_pool2d(x,kernel_size=(2,2),stride=2)
        #print(x.shape)

        x = x.view(x.size(0),-1)
        #print(x.shape)
        x = self.dropout(x)

        x=nn.functional.relu(self.bnfc1(self.fc1(x)))
        x=nn.functional.relu(self.bnfc2(self.fc2(x)))
        x= self.fc3(x)
        return x

class Params:
    def __init__(self):
        self.use_gpu = 1

        self.train = Params.Training()
        self.val = Params.Validation()
        self.test = Params.Testing()
        self.ckpt = Params.Checkpoint()
        self.optim = Params.Optimization()

    def process(self):
        self.val.vis = self.val.vis.ljust(3, '0')
        self.test.vis = self.test.vis.ljust(2, '0')
    
    class Optimization:
        """
        You can modify this class to fine-tune your model
        """
        def __init__(self):
            self.type = 'adam' # You can choose either 'adam' or 'sgd' to play around with optimizer type
            self.lr = 5e-4  # You can play around with the learning rate here
            self.momentum = 0.9
            self.eps = 1e-8
            self.weight_decay = 1e-4  # You can modify weight decay here

    class Training:
        """
        You can modify this class to change parameters for classifier training
        """
        def __init__(self):
            self.probs_data = ''
            self.batch_size = 128  # you can change batch size here
            self.n_workers = 4  
            self.n_epochs =27  # you can modify number of epoches here

    class Validation:
        def __init__(self):
            self.batch_size = 16
            self.n_workers = 1
            self.gap = 1
            self.ratio = 0.2
            self.vis = '0'
            self.tb_samples = 10

    class Testing:
        def __init__(self):
            self.enable = 0
            self.batch_size = 24
            self.n_workers = 1
            self.vis = '0'

    class Checkpoint:
        """
        :ivar load:
            0: train from scratch;
            1: load checkpoint if it exists and continue training;
            2: load checkpoint and test;

        :ivar save_criteria:  when to save a new checkpoint:
            val-acc: validation accuracy increases;
            val-loss: validation loss decreases;
            train-acc: training accuracy increases;
            train-loss: training loss decreases;

        :ivar path: path to the checkpoint
        """

        def __init__(self):
            self.load = 1
            self.path = './checkpoints/model.pt'
            self.save_criteria = [
                'train-acc',
                'train-loss',
                'val-acc',
                'val-loss',
            ]