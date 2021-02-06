import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import seaborn as sns
sns.set_palette('hls', 10)
import matplotlib.cm as cm

class HopfieldNetwork(object):      
    def train_weights(self, train_data):
        print("Start to train weights...")
        num_data =  len(train_data)
        self.num_neuron = train_data[0].shape[0]
        
        # initialize weights
        W = np.zeros((self.num_neuron, self.num_neuron))
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data*self.num_neuron)
        
        # Hebb rule
        for i in tqdm(range(num_data)):
            t = train_data[i] - rho
            W += np.outer(t, t)
        
        # Make diagonal element of W into 0
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data
        
        self.W = W 
    
    def predict(self, data, num_iter=20, threshold=0, asyn=False):
        print("Start to predict...")
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn
        
        # Copy to avoid call by reference 
        copied_data = np.copy(data)
        
        # Define predict list
        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(copied_data[i]))
        return predicted
    
    def _run(self, init_s):
        if self.asyn==False:
            """
            Synchronous update
            """
            # Compute initial state energy
            s = init_s

            e = self.energy(s)
            
            # Iteration
            for i in range(self.num_iter):
                # Update s
                s = np.sign(self.W @ s - self.threshold)
                # Compute new state energy
                e_new = self.energy(s)
                
                # s is converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s
        else:
            """
            Asynchronous update
            """
            # Compute initial state energy
            s = init_s
            e = self.energy(s)
            
            # Iteration
            for i in range(self.num_iter):
                for j in range(100):
                    # Select random neuron
                    idx = np.random.randint(0, self.num_neuron) 
                    # Update s
                    s[idx] = np.sign(self.W[idx].T @ s - self.threshold)
                
                # Compute new state energy
                e_new = self.energy(s)
                
                # s is converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s
    
    
    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()

def parser(filename):
    weights = []
    cnt = 0
    with open(filename) as file_Obj:
        strs = []
        for line in file_Obj:
            if cnt == 12:
                data = np.array(strs)
                data = np.where(data == " " ,-1,1)
                weights.append(data)
                strs = []
                cnt = 0
            else:
                lines = []
                if line[-1] == '\n':
                    lines = list(line[:-1])
                else:
                    lines = list(line)
                cnt = cnt + 1
                strs.append(lines)
        data = np.array(strs)
        data = np.where(data == " " ,-1,1)
        weights.append(data)
    return weights

def bonus_parser(filename):
    weights = []
    cnt = 0
    with open(filename) as file_Obj:
        strs = []
        for line in file_Obj:
            if cnt == 10:
                data = np.array(strs)
                data = np.where(data == " " ,-1,1)
                weights.append(data)
                strs = []
                cnt = 0
            else:
                lines = []
                if line[-1] == '\n':
                    lines = list(line[:-1])
                else:
                    lines = list(line)
                cnt = cnt + 1
                strs.append(lines)
        data = np.array(strs)
        data = np.where(data == " " ,-1,1)
        weights.append(data)
    return weights

def preprocessing(data):
    if data.shape[0] > data.shape[1]:
        negative = np.zeros((int(data.shape[0]), int(data.shape[0] - data.shape[1]) ))
        data = np.hstack([data,negative])
    else:
        negative = np.zeros((int(data.shape[1] - data.shape[0]) , int(data.shape[1])))
        data = np.vstack([data,negative])
    data[data == 0] = -1
    return data

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def plot(data, test, predicted, figsize=(12, 10)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i==0:
            axarr[i, 0].set_title('train data')
            axarr[i, 1].set_title("test data")
            axarr[i, 2].set_title('Predict')

        axarr[i, 0].imshow(np.where(data[i]<1, 1, 0), cmap='gray')
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(np.where(test[i]<1, 1, 0), cmap='gray')
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(np.where(predicted[i]<1, 1, 0), cmap='gray')
        axarr[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig("result.png" , dpi=80)
    

def main(train_file,test_file,threshold,asyn):
    if train_file[:5] == "basic":
        train = parser(os.path.join("train",train_file))
    else:
        train = bonus_parser(os.path.join("train",train_file))

    if test_file[:5] == "basic":
        test = parser(os.path.join("test",test_file))
    else:
        test = bonus_parser(os.path.join("test",test_file))
    for i in range(len(train)):
        train[i] = preprocessing(train[i])
        train[i] = train[i].flatten()
    for i in range(len(test)):
        test[i] = preprocessing(test[i])
        test[i] = test[i].flatten()
    model = HopfieldNetwork()
    model.train_weights(train)
    if asyn == 0:
        predicted = model.predict(test,  threshold=threshold, asyn=False)
    else:
        predicted = model.predict(test,  threshold=threshold, asyn=True)
    plot(train, test, predicted)