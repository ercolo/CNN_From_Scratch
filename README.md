# CNN From Scratch 🚀

Questo progetto ha l’obiettivo di **costruire una CNN da zero** utilizzando solo operazioni base su tensori, senza ricorrere a funzioni preconfezionate di librerie deep learning come PyTorch o TensorFlow.

L’intento è comprendere a fondo ogni componente di una rete convoluzionale, dal forward pass alla backpropagation, scrivendo manualmente ogni passo per apprendere i meccanismi interni delle CNN.

---

## 📋 Roadmap del progetto

### 1️⃣ Gestione dei Tensori  
- Rappresentare immagini (H×W×C), kernel (H_f×W_f×C_in×C_out) e bias (1D) come tensori.  
- Gestire batch (B×C×H×W), slicing e broadcasting.

### 2️⃣ Convoluzione 2D Manuale  
- Loop su pixel, canali e kernel.  
- Calcolo somma pesata + bias.

### 3️⃣ Funzione di Attivazione  
- Implementazione ReLU (max(0,x)), sigmoid, tanh.

### 4️⃣ Pooling  
- Max pooling e average pooling con stride.

### 5️⃣ Flatten + Fully Connected Layer  
- Conversione tensori in vettori.  
- Calcolo prodotto matrice-vettore + bias.

### 6️⃣ Loss Function  
- Cross-entropy con softmax per classificazione multi-classe.

### 7️⃣ Backpropagation Manuale  
- Calcolo gradiente per layer convoluzionali, attivazioni, pooling e fully connected.

### 8️⃣ Ottimizzazione  
- Stocastic Gradient Descent (SGD), momentum e weight decay.

### 9️⃣ Debugging  
- Controllo shape tensor, verifica gradienti (gradient check).

---

## 📂 Struttura del progetto


## Requisiti
- Python 3.x
- torch
- opencv-python
- matplotlib

## Come si esegue

```bash
python src/input.py# CNN_From_Scratch