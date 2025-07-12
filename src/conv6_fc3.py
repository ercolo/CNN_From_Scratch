import torch
import torch.nn.functional as F
import cv2 as cv
import os
import matplotlib.pyplot as plt
import math
import random
import kornia.augmentation as K

class model():
    def __init__(self):
        self.learning_rate = 0.001
        self.kernelh , self.kernelw = 3,3
        self.batch,self.label,self.classi = self.batchInit()
        self.batch = self.batch.to("cuda")
        self.label = torch.tensor(self.label,dtype=torch.long).to("cuda")
        self.C_in,self.batch_size = self.batch.shape[1],self.batch.shape[0]
        
        #Inizializzazione pesi per la convoluzione
        self.w1_conv = torch.randn(64,self.C_in,self.kernelh,self.kernelw)*math.sqrt(2/(self.C_in*self.kernelh*self.kernelw))
        self.w1_conv = self.w1_conv.to("cuda").detach().requires_grad_()
        
        self.w2_conv = torch.randn(64,64,self.kernelh,self.kernelw)*math.sqrt(2/(64*self.kernelh*self.kernelw))
        self.w2_conv = self.w2_conv.to("cuda").detach().requires_grad_()
        
        self.w3_conv = torch.randn(128,64,self.kernelh,self.kernelw)*math.sqrt(2/(64*self.kernelh*self.kernelw))
        self.w3_conv = self.w3_conv.to("cuda").detach().requires_grad_()
        
        self.w4_conv = torch.randn(128,128,self.kernelh,self.kernelw)*math.sqrt(2/(128*self.kernelh*self.kernelw))
        self.w4_conv = self.w4_conv.to("cuda").detach().requires_grad_()
        
        self.w5_conv = torch.randn(256,128,self.kernelh,self.kernelw)*math.sqrt(2/(128*self.kernelh*self.kernelw))
        self.w5_conv = self.w5_conv.to("cuda").detach().requires_grad_()
        
        self.w6_conv = torch.randn(256,256,self.kernelh,self.kernelw)*math.sqrt(2/(256*self.kernelh*self.kernelw))
        self.w6_conv = self.w6_conv.to("cuda").detach().requires_grad_()
        
        #Inizializzazione bias per la convoluzione
        self.b1_conv = torch.zeros(64).to("cuda").detach().requires_grad_()
        
        self.b2_conv = torch.zeros(64).to("cuda").detach().requires_grad_()
        
        self.b3_conv = torch.zeros(128).to("cuda").detach().requires_grad_()
        
        self.b4_conv = torch.zeros(128).to("cuda").detach().requires_grad_()
        
        self.b5_conv = torch.zeros(256).to("cuda").detach().requires_grad_()
        
        self.b6_conv = torch.zeros(256).to("cuda").detach().requires_grad_()
        #Inizializzazione weight FC Layer
        self.shape_flattern = int(((((self.batch.shape[2])//2)//2)//2)*((((self.batch.shape[3])//2)//2)//2)*self.w6_conv.shape[0])

        self.fc1_weight = torch.randn(2048,self.shape_flattern)*math.sqrt(2/self.shape_flattern)
        self.fc1_weight = self.fc1_weight.to("cuda").detach().requires_grad_()
        
        self.fc2_weight = torch.randn(1024,self.fc1_weight.shape[0])*math.sqrt(2/self.fc1_weight.shape[0])
        self.fc2_weight = self.fc2_weight.to("cuda").detach().requires_grad_()
        
        self.fc3_weight = torch.randn(self.classi,self.fc2_weight.shape[0])*math.sqrt(2/self.fc2_weight.shape[0])
        self.fc3_weight = self.fc3_weight.to("cuda").detach().requires_grad_()
        
        #Inizializzazione bias FC Layer
        self.fc1_bias = torch.zeros(self.fc1_weight.shape[0]).to("cuda").detach().requires_grad_()
        self.fc2_bias = torch.zeros(self.fc2_weight.shape[0]).to("cuda").detach().requires_grad_()
        self.fc3_bias = torch.zeros(self.fc3_weight.shape[0]).to("cuda").detach().requires_grad_()
        self.aug_list = K.AugmentationSequential(
            K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            K.RandomHorizontalFlip(p=1.0),
            K.ImageSequential(K.RandomHorizontalFlip(p=1.0)),
            K.ImageSequential(K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)),
            same_on_batch=False,
            random_apply=10,
            )
    def Train(self,minibatch):
        self.size = 400
        self.start_batch = minibatch*self.size
        self.end_batch = (minibatch+1)*self.size
        self.forward()
        self.backward()
        self.update()
    def forward(self):
        self.dataAugm = self.aug_list(self.batch[self.start_batch:self.end_batch])
        self.conv1 = F.conv2d(self.dataAugm,self.w1_conv,self.b1_conv,padding=1)
        self.conv1_ReLU = F.relu(self.conv1)
        
        self.conv2 = F.conv2d(self.conv1_ReLU,self.w2_conv,self.b2_conv,padding=1)
        self.conv2_ReLU = F.relu(self.conv2)
        
        self.maxPooling1,self.index = F.max_pool2d(self.conv2_ReLU,kernel_size=(2,2),return_indices=True)
        
        self.conv3 = F.conv2d(self.maxPooling1,self.w3_conv,self.b3_conv,padding=1)
        self.conv3_ReLU = F.relu(self.conv3)
        
        self.conv4 = F.conv2d(self.conv3_ReLU,self.w4_conv,self.b4_conv,padding=1)
        self.conv4_ReLU = F.relu(self.conv4)
        
        self.maxPooling2,self.index2 = F.max_pool2d(self.conv4_ReLU,kernel_size=(2,2),return_indices=True)
        
        self.conv5 = F.conv2d(self.maxPooling2,self.w5_conv,self.b5_conv,padding=1)
        self.conv5_ReLU = F.relu(self.conv5)
        
        self.conv6 = F.conv2d(self.conv5_ReLU,self.w6_conv,self.b6_conv,padding=1)
        self.conv6_ReLU = F.relu(self.conv6)
        
        self.maxPooling3,self.index3 = F.max_pool2d(self.conv6_ReLU,kernel_size=(2,2),return_indices=True)
        
        self.flatten = self.maxPooling3.view(self.size,-1)
        
        self.fc1_layer = F.linear(self.flatten,self.fc1_weight,self.fc1_bias)
        self.fc1_layer_ReLU = F.relu(self.fc1_layer)
        self.fc1_layer_ReLU_DropOut = F.dropout(self.fc1_layer_ReLU,p=0.2)
        
        self.fc2_layer = F.linear(self.fc1_layer_ReLU_DropOut,self.fc2_weight,self.fc2_bias)
        self.fc2_layer_ReLU = F.relu(self.fc2_layer)
        self.fc2_layer_ReLU_DropOut = F.dropout(self.fc2_layer_ReLU,p=0.2)
        
        self.fc3_layer = F.linear(self.fc2_layer_ReLU_DropOut,self.fc3_weight,self.fc3_bias)
        
    def backward(self):
        self.prob = torch.softmax(self.fc3_layer,dim=1)
        self.pred = self.prob.argmax(dim=1)
        self.accuracy = (self.pred == self.label[self.start_batch:self.end_batch]).float().mean()
        self.loss = F.cross_entropy(self.fc3_layer,self.label[self.start_batch:self.end_batch])
        self.loss.backward()

    def update(self):
        with torch.no_grad():
            self.w1_conv -= self.learning_rate * self.w1_conv.grad
            self.w2_conv -= self.learning_rate * self.w2_conv.grad
            self.w3_conv -= self.learning_rate * self.w3_conv.grad
            self.w4_conv -= self.learning_rate * self.w4_conv.grad
            self.w5_conv -= self.learning_rate * self.w5_conv.grad
            self.w6_conv -= self.learning_rate * self.w6_conv.grad
            
            self.b1_conv -= self.learning_rate * self.b1_conv.grad
            self.b2_conv -= self.learning_rate * self.b2_conv.grad
            self.b3_conv -= self.learning_rate * self.b3_conv.grad
            self.b4_conv -= self.learning_rate * self.b4_conv.grad
            self.b5_conv -= self.learning_rate * self.b5_conv.grad
            self.b6_conv -= self.learning_rate * self.b6_conv.grad
            
            self.fc1_weight -= self.learning_rate * self.fc1_weight.grad
            self.fc2_weight -= self.learning_rate * self.fc2_weight.grad
            self.fc3_weight -= self.learning_rate * self.fc3_weight.grad
            
            self.fc1_bias -= self.learning_rate * self.fc1_bias.grad
            self.fc2_bias -= self.learning_rate * self.fc2_bias.grad
            self.fc3_bias -= self.learning_rate * self.fc3_bias.grad
            
            self.w1_conv.grad.zero_()
            self.w2_conv.grad.zero_()
            self.w3_conv.grad.zero_()
            self.w4_conv.grad.zero_()
            self.w5_conv.grad.zero_()
            self.w6_conv.grad.zero_()
            
            self.b1_conv.grad.zero_()
            self.b2_conv.grad.zero_()
            self.b3_conv.grad.zero_()
            self.b4_conv.grad.zero_()
            self.b5_conv.grad.zero_()
            self.b6_conv.grad.zero_()
            
            self.fc1_weight.grad.zero_()
            self.fc2_weight.grad.zero_()
            self.fc3_weight.grad.zero_()
            
            self.fc1_bias.grad.zero_()
            self.fc2_bias.grad.zero_()
            self.fc3_bias.grad.zero_()
    def Predict(self):
        pass
    def batchInit(self):
        img = []
        label = []
        path = "CNN_From_Scratch\\img\\tiny-imagenet-200\\train"
        lista_img = os.listdir(path)
        classi = len(lista_img)
        for index_element,element in enumerate(lista_img):
            classe = os.path.join(path,element,"images")
            lista_classe = sorted(os.listdir(classe))
            for i in lista_classe[:200]:
                img.append(os.path.join(classe,i))
                label.append(index_element)
        shuffle = random.sample(range(len(img)), len(img))
        self.img_shuffled = [img[i] for i in shuffle]  
        self.label_shuffled = [label[i] for i in shuffle] 
        immagine = cv.imread(self.img_shuffled[0]) 
        batch = torch.zeros(len(self.img_shuffled),len(immagine[0][0]),len(immagine),len(immagine[0]))
        for i in range(len(self.img_shuffled)):
            immagine = cv.imread(self.img_shuffled[i])
            immagine = cv.cvtColor(immagine,cv.COLOR_BGR2RGB)
            immagine = torch.from_numpy(immagine).permute(2,0,1)/255
            batch[i] = immagine
        return batch,self.label_shuffled,classi
    def convolution(self):
        pass
if  __name__ == "__main__":
    modello = model()
    epoch = 50
    for epoch in range(epoch):
        print(f"epoch:{epoch}:")
        loss_totale=0
        accuracy_totale=0
        for minibatch in range(0,modello.batch.shape[0]//400):
            modello.Train(minibatch)
            loss_totale += modello.loss.item()
            accuracy_totale += modello.accuracy.item()*100
        print(f"Loss: {loss_totale/60}  Accuracy: {accuracy_totale/60:.2f}")