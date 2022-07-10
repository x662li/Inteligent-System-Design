# import required packages
import os
import pickle
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
from train_NLP import Preprocessing

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
# test runner class
class TestRunner:
    def __init__(self):
        pass
    
    ''' 
        prediction function
    '''   
    def predict(self, data):        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.LongTensor(data)).squeeze().detach().numpy()
        return y_pred

    ''' 
        calculate prediction accuracy
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
        load pretrained model
    '''
    def load_model(self, path):
        model = torch.jit.load(path)
        self.model = model
        
            
if __name__ == "__main__": 
    LOAD_DATA = False
    MAX_LEN = 200 # this is for preprocessing, consistant with training
    MAX_WORDS = 300 # this is for preprocessing, consistant with training
    
    pos_path = os.getcwd() + '/aclImdb/test/pos'
    neg_path = os.getcwd() + '/aclImdb/test/neg'
    model_path = './models/20659339_NLP_model.pt'
    token_path = 'tokens.pickle'
    
    print('program starts')
	# 1. Load your saved model
    test_runner = TestRunner()
    test_runner.load_model(model_path)
    
	# 2. Load your testing data
    if LOAD_DATA:
        with open('./data/X_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open('./data/Y_test.pkl', 'rb') as f:
            Y_test = pickle.load(f)
        print('data loaded')
    else: # if does not load data, pre-processing again, same as training
        # load tokenizer fitted in training
        with open(token_path, 'rb') as f:
            tokens = pickle.load(f)
        # create preprocessor with pre-fitted tokenizer
        preprocesser = Preprocessing(max_len = MAX_LEN, max_words = MAX_WORDS, tokens=tokens)
        pos_list = preprocesser.load_data(pos_path)
        neg_list = preprocesser.load_data(neg_path)
        test_tot = preprocesser.create_datasets(pos_list, neg_list, shuffle=False)
        X_test = preprocesser.encode_text(test_tot['text'].values)
        Y_test = list(test_tot['target'].values)
        # save pre-processed data
        with open('./data/X_test.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        with open('./data/Y_test.pkl', 'wb') as f:
            pickle.dump(Y_test, f)
        print('pre-processing done')

	# 3. Run prediction on the test data and print the test accuracy
    print('prediction starts...')
    predictions = test_runner.predict(X_test)
    accuracy = test_runner.calc_accuracy(targets=Y_test, predictions=predictions)
    print('The testing accuracy is: ' + str(accuracy))
    