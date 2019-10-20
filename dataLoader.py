import os
import numpy as np
import pandas as pd
from pandas import ExcelFile
from OneHotEncoder import OneHotEncoder

import torch
import torch.nn as nn

class IOLDataset:
    def __init__(self, pandas_df, NaN = 999, verbose = False, max_length = 5):
        assert (type(pandas_df) == pd.core.frame.DataFrame), "only accepts pandas DF objects"
        self.df = pandas_df
        self.df.replace('NaN', NaN)
        self.patient_num = pandas_df.shape[0]
        self.verbose = verbose
        self.max_length = max_length

    '''
    Categorical vector encoding for metadata and outcome
    '''
    def one_hot_encoder(self, categorical_vector, encoded_dict={}):
        class_dict = encoded_dict
        class_index = 0
        index_list = []
        for i in range(len(categorical_vector)):
            if categorical_vector[i] not in class_dict:
                class_dict[categorical_vector[i]] = class_index
                class_index += 1
            index_list.append(class_dict[categorical_vector[i]])
        index_list = np.array(index_list)
        # change the column dimension of the one hot vector to be the same as the class dict
        one_hot_vector = np.zeros((index_list.size, len(class_dict)+1))
        one_hot_vector[np.arange(index_list.size),index_list] = 1
        return one_hot_vector, class_dict

    def set_metadata(self, metadata):
        encoded_matrix = []
        if self.verbose:
            print("Patient metadata includes: {}".format(metadata))
        for column in metadata:
            one_hot_matrix, _ = self.one_hot_encoder(self.df[column].values, {})
            encoded_matrix = one_hot_matrix if len(encoded_matrix) == 0 else np.hstack((encoded_matrix, one_hot_matrix))
            if self.verbose:
                print("{} | key pair: {}".format(column, _))
        self.metadata = encoded_matrix

    def set_outcome(self, outcome):
        encoded_matrix, _ = self.one_hot_encoder(self.df[outcome].values)
        if self.verbose:
            print("Outcome defined by: {}".format(outcome))
            print("{} | key pair: {}".format(outcome, _))
        self.outcome = encoded_matrix

    def set_delivery_time(self):
        self.delivery_time = self.df['Delivery Time'].values

    '''
    numpy datetime difference calculation is giving weird values. need to debug.
    '''
    def time_to_delivery(self, current_time):
        assert (type(current_time) == pd.core.series.Series), "{} is not pandas series".format(type(current_time))
        return np.array(self.delivery_time - current_time.values, dtype=int)

    '''
    Encoding action series
    '''
    def dict_encoder(self, categorical_vector, encoded_dict={}):
        class_dict = encoded_dict
        class_index = 0
        for i in range(len(categorical_vector)):
            if categorical_vector[i] not in class_dict:
                class_dict[categorical_vector[i]] = class_index
                class_index += 1
        return class_dict

    def run_dict_encoder(self, actions_to_encode, max_length):
        self.encoded_dict = {}
        for item in actions_to_encode:
            encoded_dict = {}
            for t in range(1, max_length + 1):
                encoded_dict = self.dict_encoder(self.df[item + str(t)].values, encoded_dict)
            self.encoded_dict[item] = encoded_dict

    def set_induction_action(self, action_series):
        base = action_series[1:]
        self.induction_series = {}
        # iterate through all time points for each action base to create a dict with complete encodings
        self.run_dict_encoder(actions_to_encode = base, max_length = self.max_length)
        # iterate through time points to gather encodings and combien with time to delivery
        for t in range(1, self.max_length + 1):
            time_to_delivery = self.time_to_delivery(self.df[action_series[0] + str(t)])
            encoded_matrix = np.expand_dims(time_to_delivery, axis = 1) # make column vector
            for item in base:
                one_hot_matrix, _ = self.one_hot_encoder(self.df[item + str(t)].values, self.encoded_dict[item])
                encoded_matrix = np.hstack((encoded_matrix, one_hot_matrix))
            self.induction_series[t] = encoded_matrix
        if self.verbose:
            print("Induciton actions include: {}".format(base))
            print("Key pair: {}".format(self.encoded_dict))

    '''
    Create dataset that rolls out the time points with one hot vector encodings and appropritate labels
    '''
    def create_dataset(self, bishop_included = True):
        self.set_metadata(['Age', 'Ethnicity category', 'Gravida', 'Parity']) # 'Current_Pregnancy_Concerns'
        self.set_outcome('Mode of Delivery')
        self.set_delivery_time()
        self.set_induction_action(['T', 'Indication', 'Action', 'PriorROM', 'Bishop'] if bishop_included else ['T', 'Indication', 'Action', 'PriorROM'])
        dataset = []
        for t in range(1, self.max_length + 1):
            action = self.induction_series[t][:,1:]
            time_to_delivery = self.induction_series[t][:,0]
            accumulated_action = action if t == 1 else accumulated_action + action
            # The above addition doesn't work cause different induction time points have
            # different number of classes, but i didn't consider that when coding the one_hot_encodings...
            # need to debug. FFS -woochan
            for patient_id in range(self.patient_num):
                dataset.append({'id':{'patient_id':patient_id, 'series':t},
                                'metadata':self.metadata[patient_id],
                                'action':accumulated_action[patient_id],
                                'time_to_delivery':time_to_delivery[patient_id],
                                'outcome':self.outcome[patient_id]})
        return dataset



class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return

    def __len__(self):
        return self.dataset_size
