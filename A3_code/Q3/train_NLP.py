# import required packages
import os
import pickle
from unicodedata import bidirectional
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
# preprocessor, load data, tokenization, text to sequence and train-test split
class Preprocessing:
    
    def __init__(self, max_len, max_words, tokens = None):
        self.max_len = max_len # max length of sequence
        self.max_words = max_words # max words considered in tokenizer
        self.tokens = tokens # tokenizer
    
    ''' 
        load data from folder, return a single list of strings
    '''
    def load_data(self, path):
        file_list = os.listdir(path) 
        text_list = []
        for file in file_list:
            with open(path + '/' + file, 'r') as f:
                text_list.append(f.readline())
        return text_list
    
    ''' 
        create pandas dataframe from loaded data, combine positive and negative list and create target column
    '''
    def create_datasets(self, pos_list, neg_list, shuffle = True):
        pos_df = pd.DataFrame(list(zip(pos_list, [1]*len(pos_list))), columns = ['text', 'target'])
        neg_df = pd.DataFrame(list(zip(neg_list, [0]*len(neg_list))), columns = ['text', 'target'])
        data_tot = pd.concat([pos_df, neg_df])
        if shuffle: # shuffle the dataset, mix positive and negative
            return  data_tot.sample(frac=1).reset_index(drop=True)
        else:
            return data_tot
    
    ''' 
        encode text with text-to-sequence, output a sequence of intergers
    '''
    def encode_text(self, data, pad=True):
        if self.tokens == None:
            self.tokens = Tokenizer(num_words=self.max_words) # create a tokenizer with max words
            self.tokens.fit_on_texts(data) # fit tokenizer on text
        if pad:
            seq = self.tokens.texts_to_sequences(data) # transform text to sequence with tokenizer
            return pad_sequences(seq, maxlen=self.max_len) # pad to max_len with 0
        else:
            return self.tokens.texts_to_sequences(data)
    
    ''' 
        train test split with ratio of validation set
    ''' 
    def train_valid_split(self, data, target, valid_ratio):
        return train_test_split(data, target, test_size = valid_ratio)
    
    ''' 
        save fitted tokenizer for testing set
    '''
    def save_tokens(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
    
# Model class
class Model(nn.Module):
        
    def __init__(self, params):
        super(Model, self).__init__()
        
        self.max_words = params['max_words'] # max words to consider for embedding
        self.emb_dim = params['emb_dim'] # output dimension for embedding
        
        self.embedding = nn.Embedding(num_embeddings=self.max_words,
                                    embedding_dim=self.emb_dim,
                                    padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.emb_dim,
                            hidden_size=self.emb_dim,
                            num_layers=1,
                            bidirectional = True,
                            batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc_1 = nn.Linear(in_features=self.emb_dim*2, out_features=self.emb_dim)
        self.fc_2 = nn.Linear(in_features=self.emb_dim, out_features=1)
    
    ''' 
        forward propagation
    '''    
    def forward(self, data):
        emb_output = self.embedding(data)
        lstm_output, _ = self.lstm(emb_output)
        drop_output = self.dropout(lstm_output)
        fc1_output = torch.relu(self.fc_1(drop_output[:,-1,:]))
        fc2_output = torch.sigmoid(self.fc_2(fc1_output))
        return fc2_output
    
# maping function for dataloader
class MapStyleDS(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class Trainer:
    def __init__(self, model, params):
        self.model = model # model to be trained
        self.batch_size = params['batch_size'] # batch size for minibatch
        self.lr = params['lr'] # learning rate
        self.num_epoch = params['num_epoch'] # number of epoch
        self.train_loader = None
        self.valid_loader = None
        self.train_accs = []
        self.valid_accs = []
        self.train_loss = []
        
    ''' 
        training method
    '''
    def train(self, X_train, Y_train, X_valid, Y_valid):
        # create train and validation dataloader
        train_set = MapStyleDS(X_train, Y_train)
        valid_set = MapStyleDS(X_valid, Y_valid)
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=False)
        # optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        # training process
        for epoch in range(self.num_epoch):
            self.model.train() # turn on traning mode
            train_preds = []
            train_y= []
            for batch_x, batch_y in self.train_loader:
                x = batch_x.type(torch.LongTensor)
                y = batch_y.type(torch.Tensor)
                y_pred = self.model(x).squeeze()
                loss = nn.functional.binary_cross_entropy(y_pred, y)
                optimizer.zero_grad() # stop recording gradient info
                loss.backward()
                optimizer.step()
                train_preds.extend(y_pred.squeeze().detach().numpy())
                train_y.extend(batch_y.squeeze().detach().numpy())
            
            valid_preds = self.validation()
            # compute train, test accuracy
            train_acc = self.calc_accuracy(train_y, train_preds)
            valid_acc = self.calc_accuracy(Y_valid, valid_preds)
            # record accuracy and loss values
            self.train_accs.append(train_acc)
            self.valid_accs.append(valid_acc)
            self.train_loss.append(loss.item())
            
            print('Epoch: %i, loss: %f, train_accuracy: %f, valid_accuracy: %f' % (epoch+1, loss.item(), train_acc, valid_acc))
        
        print('final training loss: %f, accuracy: %f.' %(loss.item(), train_acc))
    
    ''' 
        run model on validation set
    '''
    def validation(self):
        prediction = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, _ in self.valid_loader:
                x = batch_x.type(torch.LongTensor)
                y_pred = self.model(x)
                prediction.extend(y_pred.squeeze().detach().numpy())
        return prediction
    
    ''' 
        calcualte prediction accuracy
    '''
    def calc_accuracy(self, targets, predictions):
        # accuracy = true_posivie + true_negative / total_data_num
        true_pos = 0
        true_neg = 0
        
        for i in range(len(targets)):
            if (predictions[i] > 0.5) & (targets[i] == 1):
                true_pos += 1
            elif (predictions[i] < 0.5) & (targets[i] == 0):
                true_neg += 1
        return (true_pos + true_neg) / len(targets)

    ''' 
        plot accuracy plots
    '''
    def plot_acc(self):
        epochs = range(self.num_epoch)
        plt.plot(epochs, self.train_accs, label = "train_acc")
        plt.plot(epochs, self.valid_accs, label = 'valid_acc')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()
    
    '''
        save model in path
    '''
    def save_model(self, path):
        model_scripted = torch.jit.script(self.model)
        model_scripted.save(path)
            

if __name__ == '__main__':
    
    LOAD_DATA = False
    MAX_LEN = 200
    MAX_WORDS = 300
    EMB_DIM = 64
    TEST_RATIO = 0.2
    BATCH_SIZE = 100
    NUM_EPOCH = 25
    LR = 0.5
    
    pos_path = os.getcwd() + '/aclImdb/train/pos'
    neg_path = os.getcwd() + '/aclImdb/train/neg'
    save_path = './models/20659339_NLP_model.pt'
    token_path = 'tokens.pickle'
    
    print('program starts')
    
    # 1. load your training data
    # LOAD previously saved datasets
    if LOAD_DATA:
        with open('./data/X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open('./data/X_valid.pkl', 'rb') as f:
            X_valid = pickle.load(f)
        with open('./data/Y_train.pkl', 'rb') as f:
            Y_train = pickle.load(f)
        with open('./data/Y_valid.pkl', 'rb') as f:
            Y_valid = pickle.load(f)
        print('data loaded')
    else: # run preprocess
        preprocesser = Preprocessing(max_len = MAX_LEN, max_words = MAX_WORDS)
        
        pos_list = preprocesser.load_data(pos_path)
        neg_list = preprocesser.load_data(neg_path)
        train_tot = preprocesser.create_datasets(pos_list, neg_list, shuffle=True)
        encoded_traintot = preprocesser.encode_text(train_tot['text'].values)
        preprocesser.save_tokens(token_path) # save token fitted on training set
        X_train, X_valid, Y_train, Y_valid = preprocesser.train_valid_split(encoded_traintot, list(train_tot['target'].values), TEST_RATIO)
        # save
        with open('./data/X_train.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open('./data/X_valid.pkl', 'wb') as f:
            pickle.dump(X_valid, f)
        with open('./data/Y_train.pkl', 'wb') as f:
            pickle.dump(Y_train, f)
        with open('./data/Y_valid.pkl', 'wb') as f:
            pickle.dump(Y_valid, f)
        
        print('pre-processing done')

    # 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy
    print('model, trainer created')
    model = Model({'max_words': MAX_WORDS, 'emb_dim': EMB_DIM})
    trainer = Trainer(model, {
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'num_epoch': NUM_EPOCH
    })
    
    print('training start...')
    
    trainer.train(X_train, Y_train, X_valid, Y_valid)
    trainer.plot_acc()
    
    print('model training finished, save model...')
    
    # 3. Save your model
    trainer.save_model(save_path)
    
    print('model saved, training complete, please proceed to testing.')
