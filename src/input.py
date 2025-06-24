import torch
import cv2 as cv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

#Ottenere numero elementi in img
current_fd = f"CNN_From_Scratch\\img\\tiny-imagenet-200\\train"
dir_list = os.listdir(current_fd)
yTrain = []

immagini = [] 

immagini = []
for class_ind,i in enumerate(dir_list):
    path_img = os.path.join(current_fd, i, "images")
    files = os.listdir(path_img)
    for f in files:
        immagini.append(os.path.join(path_img, f))
        yTrain.append(class_ind)
print("ottenimento path immagini finito")

yTrain = torch.tensor(yTrain).to("cuda")

elementi_dir = 4   #500*200

#Preparazione TENSOR
h = 64
w = 64


#inzializzazione batch
batch = torch.zeros(elementi_dir, 3, h+2, w+2).to("cuda")
batch_conv_out = torch.zeros(elementi_dir, 16, h, w).to("cuda")     #output della batch = elementi_dir,kernel_conv,righe,colonne
batch_pool_out = torch.zeros(elementi_dir, 16, h//2, w//2).to("cuda")     #output della batch = elementi_dir,kernel_conv,righe,colonne
batch_flattern_out = torch.zeros(elementi_dir,16*(h//2)*(w//2))


#Iniializzazione kernel_conv
C_out = 16          #kernel_conv in utilizzo
C_in_conv = 3            #canali input
H_f_conv, W_f_conv = 3, 3     #grandezza kernel_conv
kernel_conv = torch.randn(C_out,C_in_conv,H_f_conv, W_f_conv)* 0.01 #valori random conn forma 16,3,3,3
kernel_conv = kernel_conv.to("cuda")

#inzializzazione bias
bias = torch.randn(16)*0.01

#Inizializzazione output
hout,wout = (h+2-3)+1,(w+2-3)+1
output = torch.zeros(elementi_dir, 16, hout, wout)


#Formattazione immagine a bgr a rgb ricomposizione della lista e normalizzazione+ padding di 1
def dataSetInput(batch,img,i):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)/255
    img = F.pad(img,pad=(1, 1, 1, 1), mode='constant', value=0)
    batch[i] = img
    
#Convoluzione
def convolution(batch,elementi_dir,bias,batch_conv_out,h,w):
    kernel_conv_img = torch.zeros(elementi_dir, 16, h, w).to("cuda")
    for i in range(0,elementi_dir,4):
        for j in range(0,16):
            for k in range(0,3):
                for l in range(1,h-2,4):
                    for m in range(1,w+1):
                                kernel_conv_img[i][j][l-1][m-1] += torch.sum(batch[i][k][l-1:l+2, m-1:m+2]*kernel_conv[j][k])
                                kernel_conv_img[i][j][l][m-1] += torch.sum(batch[i][k][l:l+3, m-1:m+2]*kernel_conv[j][k])
                                kernel_conv_img[i][j][l+1][m-1] += torch.sum(batch[i][k][l+1:l+4, m-1:m+2]*kernel_conv[j][k])
                                kernel_conv_img[i][j][l+2][m-1] += torch.sum(batch[i][k][l+2:l+5, m-1:m+2]*kernel_conv[j][k])
                                
                                kernel_conv_img[i+1][j][l-1][m-1] += torch.sum(batch[i+1][k][l-1:l+2, m-1:m+2]*kernel_conv[j][k])
                                kernel_conv_img[i+1][j][l][m-1] += torch.sum(batch[i+1][k][l:l+3, m-1:m+2]*kernel_conv[j][k])
                                kernel_conv_img[i+1][j][l+1][m-1] += torch.sum(batch[i+1][k][l+1:l+4, m-1:m+2]*kernel_conv[j][k])
                                kernel_conv_img[i+1][j][l+2][m-1] += torch.sum(batch[i+1][k][l+2:l+5, m-1:m+2]*kernel_conv[j][k])
                                
                                kernel_conv_img[i+2][j][l-1][m-1] += torch.sum(batch[i+2][k][l-1:l+2, m-1:m+2]*kernel_conv[j][k])
                                kernel_conv_img[i+2][j][l][m-1] += torch.sum(batch[i+2][k][l:l+3, m-1:m+2]*kernel_conv[j][k])
                                kernel_conv_img[i+2][j][l+1][m-1] += torch.sum(batch[i+2][k][l+1:l+4, m-1:m+2]*kernel_conv[j][k])
                                kernel_conv_img[i+2][j][l+2][m-1] += torch.sum(batch[i+2][k][l+2:l+5, m-1:m+2]*kernel_conv[j][k])
                                
                                kernel_conv_img[i+3][j][l-1][m-1] += torch.sum(batch[i+3][k][l-1:l+2, m-1:m+2]*kernel_conv[j][k])
                                kernel_conv_img[i+3][j][l][m-1] += torch.sum(batch[i+3][k][l:l+3, m-1:m+2]*kernel_conv[j][k])
                                kernel_conv_img[i+3][j][l+1][m-1] += torch.sum(batch[i+3][k][l+1:l+4, m-1:m+2]*kernel_conv[j][k])
                                kernel_conv_img[i+3][j][l+2][m-1] += torch.sum(batch[i+3][k][l+2:l+5, m-1:m+2]*kernel_conv[j][k])
                        #print(kernel_conv_img[l-1][m-1])
                print(f"colore {k}")
            batch_conv_out[i][j] = kernel_conv_img[i][j]+bias[j]
            batch_conv_out[i+1][j] = kernel_conv_img[i+1][j]+bias[j]
            batch_conv_out[i+2][j] = kernel_conv_img[i+2][j]+bias[j]
            print("immagine:",i," kernel_conv:",j)
            
#applicazione della normalizzazione     
def normalizzation(batch):
    for i in range(batch.shape[0]):
        for j in range(batch.shape[1]):
            img_np = batch[i][j].cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            batch[i][j] = torch.tensor(img_np, device=batch.device)
    return batch


#applicazione di ReLU dove se x > 0 tiene x altrimenti 0.0    
def ReLU(batch_conv_out):
    batch_conv_out = torch.where(batch_conv_out>0,batch_conv_out,torch.tensor(0.0))
    return batch_conv_out


#Max pooling della batch in un tensor di Batch,kernel,H/2,W/2
def maxPooling(batch_pool_out,batch_conv_out):
    kernel_pool_img = torch.zeros(elementi_dir, 16, int(batch_pool_out.shape[2]//2), int(batch_pool_out.shape[3]//2)).to("cuda")
    for i in range(batch_pool_out.shape[0]):
        for j in range(batch_pool_out.shape[1]):
            for k in range(kernel_pool_img.shape[2]):
                for l in range(kernel_pool_img.shape[3]):
                    kernel_pool_img[i][j][k][l] = max(batch_conv_out[i][j][k*2][l*2],batch_conv_out[i][j][k*2+1][l*2],batch_conv_out[i][j][k*2][l*2+1],batch_conv_out[i][j][k*2+1][l*2+1])
    return kernel_pool_img


#Flatteri della batch contenente il max poolin in un tensor da Batch,Kernel*H*W
def flatter(batch_pool_out,batch_flattern_out,h,w):
    for i in range(0,batch_pool_out.shape[0]):
        for j in range(0,batch_pool_out.shape[1]):
            for k in range(0,batch_pool_out.shape[2]):
                for l in range(0,batch_pool_out.shape[3]):
                    batch_flattern_out[i][(j*(h*w))+(k*w)+(l)] = batch_pool_out[i][j][k][l]
    return batch_flattern_out

def softMax(y,p):
    for i in range(0,y.shape[0]):
        row_max = torch.max(y[i])
        exp_row = torch.exp(y[i] - row_max).to("cuda")
        sum_exp = torch.sum(exp_row).to("cuda")
        for j in range(0,y.shape[1]):
            p[i][j] = exp_row[j] / sum_exp
    return p
            
def lostFunction(yTrain,p,l,lT):
    for i in range(yTrain.shape[0]):
        l[i] = -torch.log(p[i][yTrain[i]] + 1e-8)
    lT = torch.sum(l)/yTrain.shape[0]
    return l,lT

#Prendere elementi e inseririli nel tensor batch
for i in range(0,elementi_dir):
    img = cv.imread(immagini[i])
    dataSetInput(batch,img,i)
print("ottenimento immagini finito")


#Convoluzione
convolution(batch,elementi_dir,bias,batch_conv_out,h,w)

batch_conv_out = normalizzation(batch_conv_out)

batch_conv_out = ReLU(batch_conv_out)

batch_pool_out = maxPooling(batch_pool_out,batch_conv_out)

batch_flattern_out = flatter(batch_pool_out,batch_flattern_out,h//2,w//2)

for i in range(0,elementi_dir):
    for j in range(batch_conv_out.shape[1]):
        img_np = batch_conv_out[i][j].cpu().numpy()
        plt.title("conv")
        plt.imshow(img_np, cmap='gray') 
        plt.margins(0,0)
        plt.axis('off')                  
        plt.show()

for i in range(0,elementi_dir):
    for j in range(batch_pool_out.shape[1]):
        img_np = batch_pool_out[i][j].cpu().numpy()
        plt.title("max pooling")
        plt.imshow(img_np, cmap='gray') 
        plt.margins(0,0)
        plt.axis('off')                  
        plt.show()
        
#FC Layer inizializzazione variabili
X = batch_flattern_out.to("cuda")                              #batch_flattern_out = torch.zeros(elementi_dir,16*(h//2)*(w//2))    numero immagini e immagini tutte in un tensor in lunghezza
weights = torch.randn(16*(h//2)*(w//2),200)*0.01    #lunghezza immagini su una sola linea e 200 cioe le classi possibili(200)/neuroni
weights = weights.to("cuda") 
bias_FC = torch.randn(200)*0.01                     #classi possibili 200/neuroni
bias_FC = bias_FC.to("cuda") 
y = torch.zeros(elementi_dir,200).to("cuda")              #quantita immagini e neuroni/classi possibili
weights2 = torch.randn(200,200)*0.01                #lunghezza immagini su una sola linea e 200 cioe le classi possibili(200)/neuroni
weights2 = weights2.to("cuda") 
bias_FC2 = torch.randn(200)*0.01                    #classi possibili 200/neuroni
bias_FC2 = bias_FC2.to("cuda") 
y2 = torch.zeros(elementi_dir,200).to("cuda")       #quantita immagini e neuroni/classi possibili
p = torch.zeros_like(y2)
l = torch.zeros(elementi_dir).to("cuda")            #tensore contenente ogni loss individuale
lT = 0

p = 0.8                                             #Probabilità che nel drop out si 0 20%
yDropOut = (torch.rand_like(y) < p).float().to("cuda") 


def fullyConnectedLayer(X,weights,bias_FC,y):       #Funzione FC layer
    for i in range(X.shape[0]):
        for j in range(bias_FC.shape[0]):
            somma = (0).to("cuda")
            for k in range(X.shape[1]):
                somma += X[i][k]*weights[k][j]
            y[i][j] = somma+bias_FC[j]
    return y

y = fullyConnectedLayer(X,weights,bias_FC,y)        #implementazione FC layer 1

y = ReLU(y)             #implementazione ReLU

y = y*yDropOut/p        #implementazione Dropout_mask

y = normalizzation(y)

y2 = fullyConnectedLayer(y,weights2,bias_FC2,y2)    #implementazione FC layer 2

p = softMax(y2,p)

l,lT = lostFunction(yTrain,p,l,lT)

