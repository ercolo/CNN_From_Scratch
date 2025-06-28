import torch
import cv2 as cv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

#Ottenere numero elementi in img
current_fd = f"CNN_From_Scratch\\img\\tiny-imagenet-200\\train"
dir_list = os.listdir(current_fd)

train_size= 5000

yTrain = []
immagini = []
for class_ind, i in enumerate(dir_list):
    path_img = os.path.join(current_fd, i, "images")
    files = os.listdir(path_img)

    # Prendi solo i primi file ordinati quantita dei file per classe è train_size/200
    selected_files = sorted(files)  # o usa random.sample(files, 2) per 2 casuali

    for f in selected_files:
        immagini.append(os.path.join(path_img, f))
        yTrain.append(class_ind)
print("ottenimento path immagini finito")

elementi_dir = train_size   #500*200 è la grandezza totale del training pero troppo pesante per me
num_img = len(immagini)
#Preparazione TENSOR
h = 64
w = 64

#inzializzazione batch
batch = torch.zeros(elementi_dir, 3, h, w ).to("cuda")              #h+2,w+2 se faccio padding da solo

#Iniializzazione kernel_conv
C_out = 16          #kernel_conv in utilizzo(16 per il primo 32 per il 2 64 per il 3)
C_in_conv = 3            #canali input(RGB)
H_f_conv, W_f_conv = 3, 3     #grandezza kernel_conv

#Kernel per la convoluzione 1,2,3
kernel_conv1 = torch.randn(C_out, C_in_conv, H_f_conv, W_f_conv, device="cuda")*0.01
kernel_conv2 = torch.randn(C_out * 2, C_out, H_f_conv, W_f_conv, device="cuda")*0.01
kernel_conv3 = torch.randn(C_out * 4, C_out * 2, H_f_conv, W_f_conv, device="cuda")*0.01

#inzializzazione output convoluzioni
conv1_out = torch.zeros(elementi_dir, 16, h, w).to("cuda")           #output della batch = elementi_dir,kernel_conv1,righe,colonne
conv2_out = torch.zeros(elementi_dir, 32, h//2, w//2).to("cuda")     #output della batch = elementi_dir,kernel_conv2,righe//2,colonne//2    perchè è stato applicato un maxpool 2x2 e conv
conv3_out = torch.zeros(elementi_dir, 64, h//4, w//4).to("cuda")     #output della batch = elementi_dir,kernel_conv3,righe//4,colonne//4    

#inizializzazione output ReLU
ReLU1_out = torch.zeros(elementi_dir, 16, h, w).to("cuda")           #output della batch = elementi_dir,kernel_conv,righe,colonne
ReLU2_out = torch.zeros(elementi_dir, 32, h//2, w//2).to("cuda")     #output della batch = elementi_dir,kernel_conv,righe//2,colonne//2     perchè è stato applicato un maxpool 2x2 e conv
ReLU3_out = torch.zeros(elementi_dir, 64, h//4, w//4).to("cuda")     #output della batch = elementi_dir,kernel_conv,righe//4,colonne//4     perchè è stato applicato 2 maxpool 2x2 e conv

#inizializzazione output maxPooling
pool1_out = torch.zeros(elementi_dir, 16, h//2, w//2).to("cuda")     #output della batch = elementi_dir,kernel_conv,righe//2,colonne//2     perchè è stato applicato un maxpool 2x2 e conv
pool2_out = torch.zeros(elementi_dir, 32, h//4, w//4).to("cuda")     #output della batch = elementi_dir,kernel_conv,righe//4,colonne//4     perchè è stato applicato due maxpool 2x2 e conv
pool3_out = torch.zeros(elementi_dir, 64, h//8, w//8).to("cuda")     #output della batch = elementi_dir,kernel_conv,righe//8,colonne//8     perchè è stato applicato tre maxpool 2x2 e conv

batch_flattern_out = torch.zeros(elementi_dir,64*(h//8)*(w//8)).to("cuda")      #appiattimento di ogni immagine per il FC Layer immagini da 4096 pixel

#inzializzazione bias
bias_conv1 = torch.zeros(C_out).to("cuda")
bias_conv2 = torch.zeros(C_out*2).to("cuda")
bias_conv3 = torch.zeros(C_out*4).to("cuda")

#Formattazione immagine a bgr a rgb ricomposizione della lista e normalizzazione
def dataSetInput(batch,img,i):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)/255
    #img = F.pad(img,pad=(1, 1, 1, 1), mode='constant', value=0)    #fatto direttamente nella convoluzione usando la funzione unfold
    batch[i] = img
    
#Convoluzione
def convolution(batch, bias, kernel_conv):
    B, C, H, W = batch.shape
    out_channels = kernel_conv.shape[0]

    # Estrai patch (B, C*3*3, H*W)
    patches = F.unfold(batch, kernel_size=3, padding=1)  

    # (out_channels, C*3*3)
    kernel_matrix = kernel_conv.view(out_channels, -1)

    # Moltiplicazione (B, out_channels, H*W)
    out = kernel_matrix @ patches  # (B, out_channels, H*W)

    # Ricostruisci output (B, out_channels, H, W)
    out = out.view(B, out_channels, H, W)

    # Aggiungi bias
    out += bias.view(1, out_channels, 1, 1)

    return out
    #funzione da 0 con tutti for molto grezza niente controlli
    '''kernel_conv_img = torch.zeros(elementi_dir, 16, h, w).to("cuda")
    for i in range(0,elementi_dir,4):
        for j in range(0,kernel_conv.shape[0]):
            for k in range(0,kernel_conv[1]):
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
            conv1_out[i][j] = kernel_conv_img[i][j]+bias[j]
            conv1_out[i+1][j] = kernel_conv_img[i+1][j]+bias[j]
            conv1_out[i+2][j] = kernel_conv_img[i+2][j]+bias[j]
            print("immagine:",i," kernel_conv:",j)
    return conv1_out'''
            
#applicazione della normalizzazione     
def normalizzation(conv):
    mean = conv.mean(dim=(0, 2, 3), keepdim=True)
    std = conv.std(dim=(0, 2, 3), keepdim=True)
    conv = (conv - mean) / (std + 1e-5)
    return conv

def normalizzation_fc(conv):
    mean = conv.mean(dim=0, keepdim=True)  # media per feature
    std = conv.std(dim=0, keepdim=True)    # std per feature
    conv = (conv - mean) / (std + 1e-5)
    return conv

#applicazione di ReLU dove se x > 0 tiene x altrimenti 0.0    
def ReLU(conv_out):
    ReLU_out = conv_out*(conv_out>0)
    return ReLU_out


#Max pooling della batch in un tensor di Batch,kernel,H/2,W/2
def maxPooling(conv1_out):
    kernel_pool_img , index = F.max_pool2d(conv1_out, kernel_size=2,stride=2,return_indices=True)
    '''kernel_pool_img = torch.zeros(elementi_dir, 16, int(batch_pool_out.shape[2]//2), int(batch_pool_out.shape[3]//2)).to("cuda")
    for i in range(batch_pool_out.shape[0]):
        for j in range(batch_pool_out.shape[1]):
            for k in range(kernel_pool_img.shape[2]):
                for l in range(kernel_pool_img.shape[3]):
                    kernel_pool_img[i][j][k][l] = max(conv1_out[i][j][k*2][l*2],conv1_out[i][j][k*2+1][l*2],conv1_out[i][j][k*2][l*2+1],conv1_out[i][j][k*2+1][l*2+1])'''
    return kernel_pool_img,index


#Flatteri della batch contenente il max poolin in un tensor da Batch,Kernel*H*W
def flatter(pool_out):
    return pool_out.view(pool_out.size(0), -1)
    '''for i in range(0,batch_pool_out.shape[0]):
        for j in range(0,batch_pool_out.shape[1]):
            for k in range(0,batch_pool_out.shape[2]):
                for l in range(0,batch_pool_out.shape[3]):
                    batch_flattern_out[i][(j*(h*w))+(k*w)+(l)] = batch_pool_out[i][j][k][l]'''

def softMax(y):
    y_stable = y - y.max(dim=1, keepdim=True)[0]  # stabilizzazione numerica
    exp_y = torch.exp(y_stable)
    sum_exp = exp_y.sum(dim=1, keepdim=True)
    p = exp_y / sum_exp
    '''for i in range(0,y.shape[0]):
        row_max = torch.max(y[i])
        exp_row = torch.exp(y[i] - row_max).to("cuda")
        sum_exp = torch.sum(exp_row).to("cuda")
        for j in range(0,y.shape[1]):
            p[i][j] = exp_row[j] / sum_exp'''
    return p
            
def lostFunction(batch_label,p,l,lT):
    for i in range(batch_label.shape[0]):
        l[i] = -torch.log(p[i,batch_label[i]] + 1e-8)
    lT = torch.mean(l)
    return l,lT

def oneHotEncoding(batch_label,t):
    t.zero_()
    for i in range(0,batch_label.shape[0]):
        t[i,batch_label[i]] = 1
    return t

def fullyConnectedLayer(X,weights,bias_FC,y):       #Funzione FC layer
    y = torch.matmul(X,weights)+bias_FC
    
    '''for i in range(X.shape[0]):
        print("X:", X.shape[0],"")
        print("X:", X.shape[1],"")
        print("bias:", bias_FC.shape[0],"")
        for j in range(bias_FC.shape[0]):
            somma = 0
            for k in range(X.shape[1]):
                somma += X[i][k]*weights[k][j]
            y[i][j] = somma+bias_FC[j]'''
    return y

def backFC3(gradienteError3, y2ReLU, wFC3):
    dWFC3 = y2ReLU.T @ gradienteError3
    dBFC3 = torch.sum(gradienteError3, dim=0)
    gradienteError2 = gradienteError3 @ wFC3.T
    return dWFC3, dBFC3, gradienteError2

# backprop secondo FC layer
def backFC2(gradienteError2, yDropOut2, p_drop, y2ReLU, y1ReLU, wFC2):
    gradienteError2 = gradienteError2 * (y2ReLU > 0).float()
    gradienteError2 = gradienteError2 * yDropOut2 / p_drop
    dWFC2 = y1ReLU.T @ gradienteError2
    dBFC2 = torch.sum(gradienteError2, dim=0)
    gradienteError1 = gradienteError2 @ wFC2.T
    return dWFC2, dBFC2, gradienteError1

# backprop primo FC layer
def backFC1(gradienteError1, yDropOut1, p_drop, y1ReLU, X, wFC1, elementi_dir):
    gradienteError1 = gradienteError1 * (y1ReLU > 0).float()
    gradienteError1 = gradienteError1 * yDropOut1 / p_drop
    dWFC1 = X.T @ gradienteError1
    dBFC1 = torch.sum(gradienteError1, dim=0)
    gradienteErrorFlatter = gradienteError1 @ wFC1.T
    gradienteErrorPool3 = gradienteErrorFlatter.view(elementi_dir, 64, 8, 8)
    return dWFC1, dBFC1, gradienteErrorPool3

def backConv3(gradienteErrorPool3,index3,conv3_out,pool2_out,kernel_conv3):
    gradienteErrorConvReLU3 = F.max_unpool2d(gradienteErrorPool3, index3, kernel_size=2, stride=2)
    gradienteErrorConv3 = gradienteErrorConvReLU3*(conv3_out > 0).float()
    dW3 = gradienteKernel(pool2_out, gradienteErrorConv3, kernel_size=3, padding=1)
    dB3 = gradienteErrorConv3.sum(dim=(0, 2, 3))
    gradienteErrorPool2 = F.conv_transpose2d(gradienteErrorConv3, kernel_conv3, padding=1)
    #gradienteErrorPool2 = deCovoluzione(gradienteErrorConv3,kernel_conv3)
    return dW3,dB3,gradienteErrorPool2

def backConv2(gradienteErrorPool2,index2,conv2_out,pool1_out,kernel_conv2):
    gradienteErrorConvReLU2 = F.max_unpool2d(gradienteErrorPool2, index2, kernel_size=2, stride=2)
    gradienteErrorConv2 = gradienteErrorConvReLU2*(conv2_out > 0).float()
    dW2 = gradienteKernel(pool1_out, gradienteErrorConv2, kernel_size=3, padding=1)
    dB2 = gradienteErrorConv2.sum(dim=(0, 2, 3))
    gradienteErrorPool1 = F.conv_transpose2d(gradienteErrorConv2, kernel_conv2, padding=1)
    #gradienteErrorPool1 = deCovoluzione(gradienteErrorConv2,kernel_conv2)
    return dW2,dB2,gradienteErrorPool1

def backConv1(gradienteErrorPool1,index1,conv1_out,batch,kernel_conv1):
    gradienteErrorConvReLU1 = F.max_unpool2d(gradienteErrorPool1, index1, kernel_size=2, stride=2)
    gradienteErrorConv1 = gradienteErrorConvReLU1*(conv1_out > 0).float()
    dW1 = gradienteKernel(batch, gradienteErrorConv1, kernel_size=3, padding=1)
    dB1 = gradienteErrorConv1.sum(dim=(0, 2, 3))
    gradienteErrorInput = F.conv_transpose2d(gradienteErrorConv1, kernel_conv1, padding=1)
    #gradienteErrorInput = deCovoluzione(gradienteErrorConv1,kernel_conv1)
    return dW1,dB1,gradienteErrorInput

def deCovoluzione(y, kernel):
    kernelh, kernelw = kernel.shape[-2], kernel.shape[-1]  # es. 3x3
    kernel180 = torch.rot90(kernel, 2, dims=(-2, -1))      # flip 180°
    output_h = y.shape[2] + kernelh - 1
    output_w = y.shape[3] + kernelw - 1
    yNoConv = torch.zeros(y.shape[0], kernel.shape[1], output_h, output_w, device=y.device)

    print("deconvolution")

    for i in range(y.shape[0]):  # batch size
        print("i:", i, " shape: ", y.shape[0])
        for k in range(kernel.shape[1]):  # input channels
            acc = torch.zeros((output_h, output_w), device=y.device)
            for j in range(y.shape[1]):  # output channels
                fmap = y[i, j].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                ker = kernel180[j, k].unsqueeze(0).unsqueeze(0)  # (1,1,Kh,Kw)

                # Per ottenere output di dimensione (H + Kh - 1, W + Kw - 1)
                conv = F.conv_transpose2d(fmap, ker).squeeze(0).squeeze(0)  # (H+Kh-1, W+Kw-1)

                acc += conv
            yNoConv[i, k] = acc
    print("fine deconvolution")
    return yNoConv

def gradienteKernel(batch,conv_g,kernel_size=3,padding=1):
    """
    batch: input originale (B, C, H, W)
    conv_g: gradiente della loss rispetto all'output della conv (B, O, H_out, W_out)
    """
    B,C_in,H,W = batch.shape
    C_out = conv_g.shape[1]
    # Estrai le patch: (elementi_dir, C * 3 * 3, num_patches)
    patches = F.unfold(batch, kernel_size = kernel_size, padding=padding)   # [B, C_in*k*k, N]
    
    k_out = conv_g.view(B, C_out,-1)    # [B, C_out, N]

    grad_kernel = torch.zeros(C_out, C_in * kernel_size * kernel_size, device=batch.device)
    
    for i in range(B):
        grad_kernel += k_out[i] @ patches[i].T  # [C_out, C_in * k * k]

    grad_kernel = grad_kernel.view(C_out, C_in, kernel_size, kernel_size)
    
    return grad_kernel / B  # media su batch

'''for i in range(0,elementi_dir):
    for j in range(conv1_out.shape[1]):
        img_np = conv1_out[i][j].cpu().numpy()
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
        plt.show()'''
        
'''bn1 = torch.nn.BatchNorm2d(16).to("cuda")
bn2 = torch.nn.BatchNorm2d(32).to("cuda")
bn3 = torch.nn.BatchNorm2d(64).to("cuda")'''        
        
# Inizializzazione pesi e bias FC (una volta sola, fuori dal ciclo)
wFC1 = (torch.randn(64 * (h//8) * (w//8), 1024) * 0.01).to("cuda")
bFC1 = (torch.randn(1024) * 0.01).to("cuda")

wFC2 = (torch.randn(1024, 512) * 0.01).to("cuda")
bFC2 = (torch.randn(512) * 0.01).to("cuda")

wFC3 = (torch.randn(512, 200) * 0.01).to("cuda")
bFC3 = (torch.randn(200) * 0.01).to("cuda")

lr = 0.05
p_drop = 0.90  # probabilità drop out attivo (keep prob)

accuratezza = []
perdita = []
probabilita = []
yTrain = torch.tensor(yTrain).to("cuda")
for epoch in range(200):
    
    indices = torch.randperm(num_img)
    immagini = [immagini[i] for i in indices]
    yTrain = yTrain[indices]

    for minibatch in range(0, num_img, train_size):
        print(f"inizio immagini epoch {epoch} minibatch {minibatch // train_size}")
        batch_imgs = immagini[minibatch : minibatch + train_size]
        batch_label = yTrain[minibatch : minibatch + train_size]
        
        batch = torch.zeros(len(batch_imgs), 3, h, w).to("cuda")
        
        #Prendere elementi e inseririli nel tensor batch
        for i, img_path in enumerate(batch_imgs):
            img = cv.imread(img_path)
            dataSetInput(batch, img, i)
        
        indices = torch.randperm(batch.shape[0])
        batch = batch[indices]
        batch_label = batch_label[indices]
        
        #convoluzione 1
        conv1_out = convolution(batch, bias_conv1, kernel_conv1)
        conv1_out = normalizzation(conv1_out)                      #normalizzazione
        ReLU1_out = ReLU(conv1_out)                     #ReLU 1
        pool1_out,index1 = maxPooling(ReLU1_out)        #maxPooling 1
        
        #convoluzione 2
        conv2_out = convolution(pool1_out, bias_conv2, kernel_conv2)
        conv2_out = normalizzation(conv2_out)                      #normalizzazione
        ReLU2_out = ReLU(conv2_out)                     #ReLU 2
        pool2_out,index2 = maxPooling(ReLU2_out)        #maxPooling 2
        
        
        #convoluzione 3
        conv3_out = convolution(pool2_out, bias_conv3, kernel_conv3)
        conv3_out = normalizzation(conv3_out)                      #normalizzazione
        ReLU3_out = ReLU(conv3_out)                     #ReLU 3
        pool3_out,index3 = maxPooling(ReLU3_out)        #maxPooling 3
        
        
        #aggiorna X ogni epoca(input del FC)
        X = (flatter(pool3_out))

        #reset output e variabili ogni epoca
        y1 = torch.zeros(elementi_dir, 1024).to("cuda")
        y2 = torch.zeros(elementi_dir, 512).to("cuda")
        y3 = torch.zeros(elementi_dir, 200).to("cuda")
        target = torch.zeros_like(y3)
        prob = torch.zeros_like(y3)
        loss = torch.zeros(elementi_dir).to("cuda")
        gradiant = torch.zeros_like(y2)
        
        #FC layer 1
        y1 = fullyConnectedLayer(X, wFC1, bFC1, y1)
        y1 = normalizzation_fc(y1)
        y1ReLU = ReLU(y1)                                               #applicazione ReLU all'output del primo layer (elementi_dir,1024)
        yDropOut1 = (torch.rand_like(y1ReLU) < p_drop).float().to("cuda")    #dropout mask
        y1ReLU = y1ReLU * yDropOut1 / p_drop                                 #dropout mask

        #FC layer 2
        y2 = fullyConnectedLayer(y1ReLU, wFC2, bFC2, y2)
        y2 = normalizzation_fc(y2)
        y2ReLU = ReLU(y2)                                               #applicazione ReLU all'output del primo layer (elementi_dir,512)
        yDropOut2 = (torch.rand_like(y2ReLU) < p_drop).float().to("cuda")   #dropout mask
        y2ReLU = y2ReLU * yDropOut2 / p_drop                                 #dropout mask

        #FC layer 2

        y3 = fullyConnectedLayer(y2ReLU, wFC3, bFC3, y3)  
        
        #Calcolo probabilita e errore
        #prob = softMax(y3)                                                  #softmax
        #loss, lossTotale = lostFunction(batch_label, prob, loss, 0)        #loss
        prob = torch.softmax(y3, dim=1)
        target = oneHotEncoding(batch_label, target)
        lossTotale = -torch.sum(target * torch.log(prob + 1e-7)) / elementi_dir
        gradienteError3 = (prob - target) / elementi_dir                                 #gradiente errore per backprop
        
        dWFC3,dBFC3,gradienteError2 = backFC3(gradienteError3,y2ReLU,wFC3)
        dWFC2,dBFC2,gradienteError1 = backFC2(gradienteError2, yDropOut2, p_drop, y2ReLU, y1ReLU, wFC2)
        dWFC1,dBFC1,gradienteErrorPool3 = backFC1(gradienteError1, yDropOut1, p_drop, y1ReLU, X, wFC1, elementi_dir)
        dW3,dB3,gradienteErrorPool2 = backConv3(gradienteErrorPool3,index3,conv3_out,pool2_out,kernel_conv3)
        dW2,dB2,gradienteErrorPool1 = backConv2(gradienteErrorPool2,index2,conv2_out,pool1_out,kernel_conv2)
        dW1,dB1,gradienteErrorInput = backConv1(gradienteErrorPool1,index1,conv1_out,batch,kernel_conv1)
        
        #Correzione FC
        wFC3 -= lr * dWFC3
        bFC3 -= lr * dBFC3

        wFC2 -= lr * dWFC2
        bFC2 -= lr * dBFC2

        wFC1 -= lr * dWFC1
        bFC1 -= lr * dBFC1
        
        #Correzzion convolution
        kernel_conv3 -= lr * dW3
        bias_conv3   -= lr * dB3

        kernel_conv2 -= lr * dW2
        bias_conv2   -= lr * dB2

        kernel_conv1 -= lr * dW1
        bias_conv1   -= lr * dB1
        
        if epoch%40==0:
            lr *= 0.1

        '''plt.figure(figsize=(12, 4))

        plt.subplot(1, 5, 1)
        plt.title("pool1_out")
        plt.imshow(pool1_out[0][0].cpu(), cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 5, 2)
        plt.title("pool1_out")
        plt.imshow(pool1_out[0][1].cpu(), cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 5, 3)
        plt.title("pool1_out")
        plt.imshow(pool1_out[0][2].cpu(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 5, 4)
        plt.title("pool2_out")
        plt.imshow(pool2_out[0][0].cpu(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 5, 5)
        plt.title("pool3_out")
        plt.imshow(pool3_out[0][0].cpu(), cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()'''
        
        print(f"Epoch {epoch+1} - Loss totale: {lossTotale.item()}")
        pred = prob.argmax(dim=1)
        acc = (pred == batch_label).float().mean()
        print(f"Accuracy: {acc.item()*100:.2f}% minibatch {minibatch // train_size}")
        torch.cuda.empty_cache()
        accuratezza.append(f"{acc.item()*100:.2f}%")
        perdita.append(f"{lossTotale.item()}")
        probabilita.append(prob.argmax(dim=1).tolist())
    
with open("valori.csv", "w") as f:
    f.write("accuracy,loss,probabilities\n")
    for acc, loss, probs in zip(accuratezza, perdita, probabilita):
        probs_str = ";".join(str(p) for p in probs)
        f.write(f"{acc},{loss},{probs_str}\n")