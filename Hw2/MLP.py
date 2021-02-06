import numpy as np
import sys
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import os

class NeuralNetMLP(object):

    def __init__(self, n_hidden=30,n_hidden_layer=5,
                 l2=0.01, epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.n_hidden_layer = n_hidden_layer
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def relu(self,X):
        return np.maximum(X, 0)

    def relu_derivative(self,X):
        return 1. * (X > 0)

    def _onehot(self, y, n_classes):
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
           
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Compute forward propagation step"""
        x_in = X
        for i in range(len(self.network)):
            if i == len(self.network) -1:
                z = np.dot(x_in, self.network[i]['weight']) +self.network[i]['bias']
                exp_scores = np.exp(z)
                x_out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                self.network[i]['output'] = x_out
                x_in = x_out
            else:    
                z = np.dot(x_in, self.network[i]['weight']) +self.network[i]['bias']
                x_out = self.relu(z)
                self.network[i]['output'] = x_out
                x_in = x_out
            #print(node['weight'])
        return x_in

    def _compute_cost(self, y_enc, output):
        weight_hidden = 0
        for node in self.network:
            weight_hidden += np.sum(node['weight']**2)
        L2_term = (self.l2 *
                   (weight_hidden ))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        output= self._forward(X)
        y_pred = np.argmax(output, axis=1)
        #print(output)
        return y_pred

        # Sigmoid derivative
    def _sigmoid_derivative(self, sigmoid):
        return sigmoid*(1.0-sigmoid)

    def _layer(self,input_dim, output_dim):
        w = self.random.normal(loc=0.0, scale=0.1,
                                size=(input_dim, output_dim))
        b = np.zeros(output_dim)
        node = {"weight": w, # numpy of weights
                "bias" :b
                #"output": np.array(), # scalar
                #"delta":np.array(),
                #"dW": np.array() ,
                #"db": np.array()
                 } # scalar
        return node


    def fit(self, X_train, y_train):
        n_output =  np.unique(y_train)  # number of class labels
        n_output = int(n_output[len(n_output) - 1] + 1)
        n_features = X_train.shape[1]

        ########################
        # Weight initialization
        ########################

        # weights for input -> hidden
        self.network = []
        if self.n_hidden_layer == 0:
            self.network.append(self._layer(n_features , n_output))
        else:
            self.network.append(self._layer(n_features, self.n_hidden))
            for i in range(1, self.n_hidden_layer):
                self.network.append(self._layer(self.n_hidden, self.n_hidden))
            self.network.append(self._layer(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {'cost': [], 'train_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                output = self._forward(X_train[batch_idx])

                ##################
                # Backpropagation
                ##################
                transfer_derivative = self.relu_derivative # sig' = f(sig)
                n_layers = len(self.network)
                #print(n_layers)
                for k in reversed(range(n_layers)): # traverse backwards
                    if k == n_layers - 1:
                        # Difference between logits and one-hot target
                        err = self.network[k]['output'] - y_train_enc[batch_idx]
                        self.network[k]['delta'] = err
                        self.network[k]['dW'] = (self.network[k-1]['output']).T.dot(self.network[k]['delta'])
                        self.network[k]['db'] = np.sum(self.network[k]['delta'], axis=0)
                        self.network[k]['dW'] += self.l2 * self.network[k]['weight']
                    elif k==0:
                        self.network[k]['delta'] = self.network[k+1]['delta'].dot(self.network[k+1]['weight'].T) * transfer_derivative(self.network[k]['output'])
                        #print(self.network[k]['delta'])
                        self.network[k]['dW'] = np.dot(X_train[batch_idx].T, self.network[k]['delta'])
                        self.network[k]['db'] = np.sum(self.network[k]['delta'], axis=0)
                        self.network[k]['dW'] += self.l2 * self.network[k]['weight']
                    else:
                        #print('有進來xd')
                        self.network[k]['delta'] = self.network[k+1]['delta'].dot(self.network[k+1]['weight'].T) * transfer_derivative(self.network[k]['output'])
                        #sigma_h = (np.dot(sigma_out, self.w_out.T) *
                           #sigmoid_derivative_h)
                        self.network[k]['dW'] = np.dot(self.network[k-1]['output'].T, self.network[k]['delta'])
                        self.network[k]['db'] = np.sum(self.network[k]['delta'], axis=0)
                        self.network[k]['dW'] += self.l2 * self.network[k]['weight']

                for k in reversed(range(n_layers)): # traverse backwards
                    #print(k)
                    self.network[k]['weight'] -= self.eta * self.network[k]['dW']
                    self.network[k]['bias'] -= self.eta * self.network[k]['db']
            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            output = self._forward(X_train)
            
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=output)
            #print(cost)

            y_train_pred = self.predict(X_train)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)

        return self



def train_test_split(data,label,seed):
    random = np.random.RandomState(seed)
    indices = np.arange(data.shape[0])
    random.shuffle(indices)
    minibatch_size = int (data.shape[0] / 3)
    label = label.values
    cnt = 0
    train_data = np.empty(shape=[0, data.shape[1]])
    train_label = np.empty(shape=[0])
    for start_idx in range(0, indices.shape[0] - minibatch_size +
                               1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        if cnt == 0:
            test_data = data[batch_idx]
            test_label = label[batch_idx]
            cnt = cnt + 1
        else:
            train_data =  np.concatenate((train_data,data[batch_idx]),axis=0)
            train_label = np.concatenate((train_label,label[batch_idx]),axis=0)
    return train_data , train_label , test_data , test_label

def plot_decision_regions(data,label,X, y, classifier,fig,path, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    

    # plot the decision surface
    x1_min, x1_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
    x2_min, x2_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    color_key = []
    check = []
    for i in np.unique(Z):
        color_key.append(colors[i])
        ch = (Z == i)
        check.append(ch)
    for i in range(len(check)):
        Z[check[i]] = i
    color_key = tuple(color_key)
    cmap = ListedColormap(color_key)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[int(cl)],
                    #c=plt.cm.hot
                    marker=markers[int(cl)], 
                    label=cl, 
                    edgecolor='black')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(path, dpi=80)

def main_program(file,n_epochs,n_hidden,n_hidden_layer,eta,seed):

    if not os.path.exists('./images'):   #如果存在資料夾 , 刪除並重建一個
        os.mkdir('./images')
    if not os.path.exists('./result'):   #如果存在資料夾 , 刪除並重建一個
        os.mkdir('./result')

    df = pd.read_csv(os.path.join('datasets', file),sep=' ', header=None, encoding='utf-8')
    label = df[df.shape[1]-1]
    data = df.drop(columns=[df.shape[1]-1]).values
    X_train , y_train ,  X_test , y_test = train_test_split(data,label,seed = 10)
    nn = NeuralNetMLP(n_hidden=n_hidden, 
                       n_hidden_layer = n_hidden_layer,
                        l2=0.01, 
                        epochs=n_epochs, 
                        eta=eta,
                        minibatch_size=10, 
                        shuffle=True,
                        seed=seed)
    nn.fit(X_train=X_train, 
            y_train=y_train
            )

    y_train_pred = nn.predict(X_train)
    train_acc = (np.sum(y_train == y_train_pred)
            .astype(np.float) / X_train.shape[0])

    train_correct = X_train[y_train == y_train_pred]
    train_error = X_train[y_train != y_train_pred]
    format = lambda x : '%-6.4f' %x
    df_train_correct = pd.DataFrame(train_correct)
    for i in range(df_train_correct.shape[1]):
        df_train_correct.loc[:,i] = df_train_correct.loc[:,i].map(format)
    df_train_correct['label'] = y_train[y_train == y_train_pred].astype(int)
    df_train_correct['predict'] = y_train_pred[y_train == y_train_pred].astype(int)
    df_train_correct.to_csv('./result/'+file[:-4]+'_train_correct.txt', sep='\t', index=False)

    df_train_error = pd.DataFrame(train_error)
    for i in range(df_train_error.shape[1]):
        df_train_error.loc[:,i] = df_train_error.loc[:,i].map(format)
    df_train_error['label'] = y_train[y_train != y_train_pred].astype(int)
    df_train_error['predict'] = y_train_pred[y_train != y_train_pred].astype(int)
    df_train_error.to_csv('./result/'+file[:-4]+'_train_error.txt', sep='\t', index=False)
    

    #print(file , 'Train accuracy: %.2f%%' % (train_acc * 100))

    y_test_pred = nn.predict(X_test)
    test_acc = (np.sum(y_test == y_test_pred)
            .astype(np.float) / X_test.shape[0])
    #print(y_test != y_test_pred)
    test_correct = X_test[y_test == y_test_pred]
    test_error = X_test[y_test != y_test_pred]

    df_test_correct = pd.DataFrame(test_correct)
    for i in range(df_test_correct.shape[1]):
        df_test_correct.loc[:,i] = df_test_correct.loc[:,i].map(format)
    df_test_correct['label'] = y_test[y_test == y_test_pred].astype(int)
    df_test_correct['predict'] = y_test_pred[y_test == y_test_pred].astype(int)
    df_test_correct.to_csv('./result/'+file[:-4]+'_test_correct.txt', sep='\t', index=False)

    df_test_error = pd.DataFrame(test_error)
    for i in range(df_test_error.shape[1]):
        df_test_error.loc[:,i] = df_test_error.loc[:,i].map(format)
    df_test_error['label'] = y_test[y_test != y_test_pred].astype(int)
    df_test_error['predict'] = y_test_pred[y_test != y_test_pred].astype(int)
    df_test_error.to_csv('./result/'+file[:-4]+'_test_error.txt', sep='\t', index=False)


    #print(file , 'Test accuracy: %.2f%%' % (test_acc * 100))

    if X_train.shape[1] <=2 :
        fig = plt.figure(figsize=(5, 2.5))
        plot_decision_regions(data,label,X=X_train, y=y_train,classifier=nn ,fig=fig , 
        path=('images/'+file[:-4]+'_train.png'))
        
        fig = plt.figure(figsize=(5, 2.5))
        plot_decision_regions(data,label,X=X_test, y=y_test,
                        classifier=nn,fig = fig,path = ('./images/'+file[:-4]+'_test.png'))
    return train_acc , test_acc



