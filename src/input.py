import torch
import cv2 as cv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

#Ottenere numero elementi in img
current_fd = f"CNN_From_Scratch\\img\\tiny-imagenet-200\\train"
dir_list = os.listdir(current_fd)

yTrain = []
train_size= 6000
immagini = [] 

for class_ind, i in enumerate(dir_list):
    path_img = os.path.join(current_fd, i, "images")
    files = os.listdir(path_img)

    # Prendi solo i primi 2 file ordinati
    selected_files = sorted(files)[:30]  # o usa random.sample(files, 2) per 2 casuali

    for f in selected_files:
        immagini.append(os.path.join(path_img, f))
        yTrain.append(class_ind)
print("ottenimento path immagini finito")

yTrain = torch.tensor(yTrain).to("cuda")



elementi_dir = train_size   #500*200
batch_label = yTrain[0:train_size]

#Preparazione TENSOR
h = 64
w = 64


#inzializzazione batch
batch = torch.zeros(elementi_dir, 3, h, w ).to("cuda")              #h+2,w+2 se faccio padding da solo
batch_conv_out = torch.zeros(elementi_dir, 16, h, w).to("cuda")     #output della batch = elementi_dir,kernel_conv,righe,colonne
batch_pool_out = torch.zeros(elementi_dir, 16, h//2, w//2).to("cuda")     #output della batch = elementi_dir,kernel_conv,righe,colonne
batch_flattern_out = torch.zeros(elementi_dir,16*(h//2)*(w//2))


#Iniializzazione kernel_conv
C_out = 16          #kernel_conv in utilizzo
C_in_conv = 3            #canali input
H_f_conv, W_f_conv = 3, 3     #grandezza kernel_conv
kernel_conv = torch.randn(C_out,C_in_conv,H_f_conv, W_f_conv)* 0.05 #valori random conn forma 16,3,3,3
kernel_conv = kernel_conv.to("cuda")

#inzializzazione bias
bias = torch.randn(16)*0.05
bias = bias.to("cuda")
#Inizializzazione output
hout,wout = (h+2-3)+1,(w+2-3)+1
output = torch.zeros(elementi_dir, 16, hout, wout)


#Formattazione immagine a bgr a rgb ricomposizione della lista e normalizzazione+ padding di 1
def dataSetInput(batch,img,i):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)/255
    #img = F.pad(img,pad=(1, 1, 1, 1), mode='constant', value=0)
    batch[i] = img
    
#Convoluzione
def convolution(batch,elementi_dir,bias,batch_conv_out,kernel_conv,h,w):
    #funzione da 0 con unfold
    # Estrai le patch: (elementi_dir, C * 3 * 3, num_patches)
    patches = F.unfold(batch, kernel_size=(3,3), padding=1)  # shape: (B, 27, N)
    #num_patches = patches.shape[-1]

    # Appiattisci i kernel: (16, 27)
    kernel_matrix = kernel_conv.view(16, -1)

    patches = patches.permute(0, 2, 1)  # (B, N, C*3*3)

    kernel_t = kernel_matrix.t()  # (C*3*3, out_channels)

    # Moltiplicazione batch matrix
    out = torch.matmul(patches, kernel_t)  # (B, N, out_channels)

    # Trasponi per avere (B, out_channels, N)
    out = out.permute(0, 2, 1)

    # Fold per ricostruire dimensioni spaziali
    out_2d = F.fold(out, output_size=(h, w), kernel_size=1)

    # Aggiungi bias
    out_2d += bias.view(1, -1, 1, 1)   
    # Convoluzione manuale
    '''
    for b in range(elementi_dir):            # per ogni immagine del batch
        for f in range(16):                  # per ogni filtro/kernel
            for n in range(num_patches):     # per ogni patch
                for i in range(kernel_matrix.shape[1]):  # per ogni elemento (27)
                    out[b][f][n] += kernel_matrix[f][i] * patches[b][i][n]
        print(f"img{i}")'''
    print("conv")
    
    return out_2d
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
            batch_conv_out[i][j] = kernel_conv_img[i][j]+bias[j]
            batch_conv_out[i+1][j] = kernel_conv_img[i+1][j]+bias[j]
            batch_conv_out[i+2][j] = kernel_conv_img[i+2][j]+bias[j]
            print("immagine:",i," kernel_conv:",j)
    return batch_conv_out'''
            
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
    kernel_pool_img , index = F.max_pool2d(batch_conv_out, kernel_size=2,stride=2,return_indices=True)
    '''kernel_pool_img = torch.zeros(elementi_dir, 16, int(batch_pool_out.shape[2]//2), int(batch_pool_out.shape[3]//2)).to("cuda")
    for i in range(batch_pool_out.shape[0]):
        for j in range(batch_pool_out.shape[1]):
            for k in range(kernel_pool_img.shape[2]):
                for l in range(kernel_pool_img.shape[3]):
                    kernel_pool_img[i][j][k][l] = max(batch_conv_out[i][j][k*2][l*2],batch_conv_out[i][j][k*2+1][l*2],batch_conv_out[i][j][k*2][l*2+1],batch_conv_out[i][j][k*2+1][l*2+1])'''
    print("MP")
    print(kernel_pool_img.shape)
    return kernel_pool_img,index


#Flatteri della batch contenente il max poolin in un tensor da Batch,Kernel*H*W
def flatter(batch_pool_out,batch_flattern_out):
    batch_flattern_out = batch_flattern_out.view(batch_pool_out.shape[0],-1)
    '''for i in range(0,batch_pool_out.shape[0]):
        for j in range(0,batch_pool_out.shape[1]):
            for k in range(0,batch_pool_out.shape[2]):
                for l in range(0,batch_pool_out.shape[3]):
                    batch_flattern_out[i][(j*(h*w))+(k*w)+(l)] = batch_pool_out[i][j][k][l]'''
    print("FLT")
    return batch_flattern_out

def softMax(y,p):
    for i in range(0,y.shape[0]):
        row_max = torch.max(y[i])
        exp_row = torch.exp(y[i] - row_max).to("cuda")
        sum_exp = torch.sum(exp_row).to("cuda")
        for j in range(0,y.shape[1]):
            p[i][j] = exp_row[j] / sum_exp
    print("SM")
    return p
            
def lostFunction(batch_label,p,l,lT):
    for i in range(batch_label.shape[0]):
        l[i] = -torch.log(p[i][batch_label[i]] + 1e-8)
    lT = torch.sum(l)/batch_label.shape[0]
    print("LF")
    return l,lT

def oneHotEncoding(batch_label,t):
    for i in range(0,batch_label.shape[0]):
        t[i][batch_label[i]] = 1
    print("OHE")
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
    print("FC")
    return y

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

def calcoloGradianti(
    y2, p, t, g2, yr, y, X,
    w2, b2, w, b,
    yDropOut, p_drop,
    dw2, db2, dw1, db1, g,
    dk, dbk,
    bco, batch,
    height, width,
    elementi_dir,
    kernel_conv, bias,
    lr,index,bpo
):
    print("inizio backprop")
    dw2 = yr.T @ g2
    #versione semplice : dw2 = yr.T @ g2   
    '''for i in range(0,yr.shape[1]):      #yr shape(element_dir,200(neurons))
        for j in range(0,g2.shape[1]):  #g2 shape(element_dir,200)
            for k in range(0,yr.shape[0]):
                dw2[i][j] += yr[k][i] * g2[k][j]  ''' '''    
                       i
                    [1,2,3]               k
                    [4,5,6]         [1,4,7,10,13]
                k   [7,8,9]      =  [2,5,8,11,14]   i      [x1,x2,x3]        x1 = (1*1)+(4*4)+(7*7)+(10*10)+(13*13)
                    [10,12,12]      [3,6,9,12,15]          [x4,x5,x6]        x2 = (1*2)+(4*5)+(7*8)+(11*11)+(13*14)
                    [13,14,15].T                           [x1,x2,x3]        x3 = (1*3)+(4*6)+(7*9)+(11*12)+(13*15)
                                                                             x4 = (2*1)+(5*4)+(8*7)+(12*10)+(14*13)
                       j                                                     x5 = (2*2)+(5*5)+(8*8)+(12*11)+(14*14)
                    [1,2,3]                                                  x6 = (2*3)+(5*6)+(8*9)+(12*12)+(14*15)                             
                    [4,5,6]                                                  x7 = (3*1)+(6*4)+(9*7)+(13*10)+(15*13)
                k   [7,8,9]                                                  x8 = (3*2)+(6*5)+(9*8)+(13*11)+(15*14)
                    [10,12,12]                                               x9 = (3*3)+(6*6)+(9*9)+(13*12)+(15*15)
                    [13,14,15]                                               
                                                                      
                ''' 
    
    print("db2 = g2.sum(dim=0)")
    db2 = g2.sum(dim=0)
    '''for i in range(0,g2.shape[1]):      #db2 shape(200(neurons))
        for j in range(0,g2.shape[0]):   #g2 shape(element_dir,200)
            db2[i] += g2[j][i]'''

    print("g = g2 @ w2.T")
    g = g2 @ w2.T
    '''for i in range(0,g2.shape[0]):      #w2 shape(200,200(neurons))
        for j in range(0,w2.shape[0]):  #g2 shape(element_dir,200)
            for k in range(0,w2.shape[1]):
                g[i][j] += g2[i][k]*w2[k][j]'''
    g = g * (y > 0).float()
    g = g * yDropOut / p_drop

    print("dw1 = X.T @ g")
    dw1 = X.T @ g
    '''for i in range(0,X.shape[1]):      #X shape(elementi_dir,16*(h//2)*(w//2))
        print("X.shape[1]: ",X.shape[1]," i:",i)
        for j in range(0,g.shape[1]):  #g shape(element_dir,200)
            for k in range(0,X.shape[0]):
                dw1[i][j] += X[k][i] * g[k][j]    ''' 
    db1 = g.sum(dim=0)

    print("g @ w.T")
    g0 = g @ w.T
    '''g0 = torch.zeros(X)
    for i in range(0,w.shape[1]):      #w shape(16*(h//2)*(w//2),200(neurons))
        for j in range(0,g.shape[1]):  #g shape(element_dir,200)
            for k in range(0,w.shape[0]):
                g0[j][i] += g[j][k]*w[k][i]'''
    print("da flattern a normale")            
    g0c = torch.zeros(elementi_dir,16,32,32)
    g0c = g0.view(elementi_dir,16,32,32)
    '''for i in range(g0c.shape[0]):
        for j in range(g0c.shape[1]):
            for k in range(g0c.shape[2]):
                for l in range(g0c.shape[3]):
                    g0c[i][j][k][l] = g0[i][j*((height//2)*(width//2))+(k*(width//2))+l]'''
    print("max pooling")
    dbco = torch.zeros_like(bco)
    dbco = F.max_unpool2d(bpo, index, kernel_size=2, stride=2)
    print("size : ",bco.shape,"size : ",dbco.shape,"size : ",bpo.shape)
    '''
    for i in range(dbco.shape[0]):
        for j in range(dbco.shape[1]):
            for k in range(0, dbco.shape[2], 2):
                for l in range(0, dbco.shape[3], 2):
                    pool = [
                        bco[i][j][k][l],
                        bco[i][j][k+1][l],
                        bco[i][j][k][l+1],
                        bco[i][j][k+1][l+1]
                    ]
                    max_val = max(pool)
                    max_idx = pool.index(max_val)
                    pos = [(0,0), (1,0), (0,1), (1,1)][max_idx]
                    dbco[i][j][k+pos[0]][l+pos[1]] = 1.0'''
    print("calcolo pool e conv")
    g0c_upsampled = F.interpolate(g0c, scale_factor=2, mode='nearest').to("cuda")
    print("bco",bco.shape)
    g_pool = g0c_upsampled * dbco
    g_conv = g_pool * (bco > 0).float()
    print("convulzione")
    dk = torch.zeros_like(kernel_conv).to("cuda")
    bk = torch.zeros_like(bias).to("cuda")
    dk = gradienteKernel(batch,g_conv)
    dbk = g_conv.sum(dim=(0, 2, 3))
    print("ricalcolo valori w e b")
    w2 -= lr * dw2
    b2 -= lr * db2
    w -= lr * dw1
    b -= lr * db1
    kernel_conv -= lr * dk
    bias -= lr * dbk
    print("BWP")
    return w2, b2, w, b, kernel_conv, bias
    
#Prendere elementi e inseririli nel tensor batch
for i in range(0,elementi_dir):
    img = cv.imread(immagini[i])
    dataSetInput(batch,img,i)
print("ottenimento immagini finito")

'''for i in range(0,elementi_dir):
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
        plt.show()'''
        
# Inizializzazione pesi e bias FC (una volta sola, fuori dal ciclo)
weights = torch.randn(16 * (h//2) * (w//2), 200) * 0.05
weights = weights.to("cuda")
bias_FC = torch.randn(200) * 0.05
bias_FC = bias_FC.to("cuda")

weights2 = torch.randn(200, 200) * 0.05
weights2 = weights2.to("cuda")
bias_FC2 = torch.randn(200) * 0.05
bias_FC2 = bias_FC2.to("cuda")

dk = torch.zeros_like(kernel_conv)
dbk = torch.zeros_like(bias)

dw1 = torch.zeros_like(weights)
db1 = torch.zeros_like(bias_FC)
dw2 = torch.zeros_like(weights2)
db2 = torch.zeros_like(bias_FC2)

lr = 0.05
p_drop = 0.8  # probabilità drop out attivo (keep prob)

for epoch in range(15):
    # Convoluzione
    batch_conv_out = convolution(batch, elementi_dir, bias, batch_conv_out, kernel_conv, h, w)
    batch_conv_out = normalizzation(batch_conv_out)
    batch_conv_out = ReLU(batch_conv_out)
    batch_pool_out,index = maxPooling(batch_pool_out, batch_conv_out)
    batch_flattern_out = flatter(batch_pool_out, batch_flattern_out)

    # Aggiorna X ogni epoca (input FC)
    X = batch_flattern_out.to("cuda")

    # Reset output e variabili ogni epoca
    y = torch.zeros(elementi_dir, 200).to("cuda")
    y2 = torch.zeros(elementi_dir, 200).to("cuda")
    t = torch.zeros_like(y2)
    p = torch.zeros_like(y2)
    l = torch.zeros(elementi_dir).to("cuda")
    g = torch.zeros_like(y2)
    # FC layer 1
    y = fullyConnectedLayer(X, weights, bias_FC, y)
    yr = ReLU(y)

    # Dropout mask
    yDropOut = (torch.rand_like(yr) < p_drop).float().to("cuda")
    yr = yr * yDropOut / p_drop

    # FC layer 2
    y2 = fullyConnectedLayer(yr, weights2, bias_FC2, y2)

    # Softmax
    p = softMax(y2, p)

    # Loss
    l, lT = lostFunction(batch_label, p, l, 0)

    # One hot encoding label batch
    t = oneHotEncoding(batch_label, t)

    # Gradiente errore
    g2 = p - t
    # Calcolo e aggiornamento pesi e bias
    weights2, bias_FC2, weights, bias_FC, kernel_conv, bias = calcoloGradianti(
        y2, p, t, g2, yr, y, X,
        weights2, bias_FC2, weights, bias_FC,
        yDropOut, p_drop, dw2, db2, dw1, db1, g, dk, dbk,
        batch_conv_out, batch, h, w, elementi_dir,
        kernel_conv, bias,
        lr,index,batch_pool_out
    )

    print(f"Epoch {epoch+1} - Loss totale: {lT.item()}")
    pred = p.argmax(dim=1)
    acc = (pred == batch_label).float().mean()
    print(f"Accuracy: {acc.item()*100:.2f}%")
    torch.cuda.empty_cache()
