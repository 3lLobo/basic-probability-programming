'''
Created on Sep 30, 2015

@author: Philip Schulz
'''

import sys


def accuracy_checker(gold, pred):
    
    gold_labels = dict()
    pred_labels = dict()

    try:
        with open(gold) as goldFile:
            for line in goldFile:
                elements = line.split()
                gold_labels[elements[0]] = elements[1]
                
        with open(pred) as pred_file:
            for line in pred_file:
                elements = line.split()
                pred_labels[elements[0]] = elements[1]
    
    except IOError as e:
        print(e)
        print("One of the files does not exist on your computer.")
        sys.exit(0)
        
    if len(gold_labels) != len(pred_labels):
        print ('The lists are of different size. Please make sure to only ' +
        'use equally sized lists.')
        sys.exit(0)
        
    overlap = 0
    for doc, label in gold_labels.items():
        if pred_labels[doc] == label:
            overlap += 1
    
    overlap /= float(len(gold_labels))
    overlap *= 100
    
    print('The overlap between the gold list and the student output is {}%'.format(overlap))

"""
The file implementing Naive Bayes classifier.
"""
###
# You have to work on this file. Your task is to implement the methods
# marked as TODO.

from collections import Counter
import math
import numpy as np

class NaiveBayes(object):
    '''
    This class implements a naive bayes classifier.
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        # The following Counter will be used to count the number of text
        # occurrencies per label.
        self.label_counts = Counter()
        # The following dictionary will be used to map labels to Counters. Each of these Counters
        # contains the feature counts given the label.
        self.feature_counts = dict()

        # The following dictionary will be used to collect
        # prior probabilities of labels.
        self.label_probs = dict()
        # The following dictionary will be used to collect feature
        # probabilities given a label.
        self.feature_probs = dict()
        #A set that contains all words encountered during training.
        self.vocabulary = set()

    def train(self, data, label):
        '''
        Train the classifier by counting features in the data set.
        
        :param data: A stream of string data from which to extract features
        :param label: The label of the data
        '''
        features = list()
        for word in data.read().split():
            feature = ''.join(filter(str.isalpha, word)).lower()
        
            if len(feature) > 2 and len(feature) < 11:
                features.append(feature)
        if features != None:
            self.add_feature_counts(features, label)
        # for line in data:
        #     self.add_feature_counts(line.lower().split(), label)
    
    def add_feature_counts(self, features, label):
        '''
        Count the features in a feature list.
        
        :param features: a list of features.
        :param label: the label of the data file from which the features were extracted.
        '''
        # This method updates feature_counts by features given the label. It
        # should also update vocabulary with features.
        # TODO: implement this!
        if label in self.feature_counts:
            self.feature_counts[label].update(features)
        else:
            self.feature_counts[label] = Counter()
            self.feature_counts[label].update(features)
        self.vocabulary.update(features)

    def smooth_feature_counts(self, smoothing=1):
        '''Smooth the collected feature counts

        :param smoothing: The smoothing constant
        '''
        # This method smoothes counts in feature_counts. Check the assignment
        # description on how to do this.
        # Add +1 to all counters
        for label in self.feature_counts:
            for _ in range(smoothing):
                self.feature_counts[label].update(list(self.vocabulary))
        
        
    def update_label_count(self,label):
        '''
        Increase the count for the supplied label by 1.
        
        :param label: The label whose count is to be increased.
        '''
        self.label_counts.update([label])
        
    def log_normalise_label_probs(self):
        '''
        Take label counts in label_counts, normalize them to
        probabilities, transform them to logprobs and update label_probs
        with the logprobs.
        '''
        # Take label_counts, and update label_probs.
        # label_probs should have labels as keys. The values are
        # log-probability of each label. The probability is created
        # by normalizing values in label_counts, after that it is
        # log-transformed.
        norm = np.sum(list(self.label_counts.values()))
        for label in self.label_counts:
            self.label_probs[label] = np.log(float(self.label_counts[label]) / norm)

            
    def log_normalise_feature_probs(self):
        '''
        Take feature counts in feature_counts and for each label, normalize
        them to probabilities and turn them into logprobs. update
        feature_probs with the created logprobs.
        '''
        # Take feature_counts, update feature_probs.
        # feature_probs have labels as keys. The values are
        # dictionaries that have features as keys as log-probs as values.
        for label in self.feature_counts:
            self.feature_probs[label] = dict()
            label_norm = np.sum(list(self.feature_counts[label].values()))
            for word in self.vocabulary:
                self.feature_probs[label][word] = np.log(float(self.feature_counts[label][word]) / label_norm)
            

                
    def predict(self, data):
        ''' 
        Predict the most probable label according to the model on a stream of data.
        
        :param data: A stream of string data from which to extract features
        :return: the most probable label for the data (type string)
        '''
        features = [''.join(filter(str.isalpha, word)).lower() for word in data.split()]
        # features = data.lower().split(' ')
        max_prob  = 20000000
        for label in self.label_counts:
            # Bayesss
            p = - self.label_probs[label] - np.sum([self.feature_probs[label][word] for word in features if word in self.vocabulary])
            if p < max_prob:
                max_prob = p
                mp_label = label
        return mp_label

'''
Created on Sep 23, 2015

@author: Philip Schulz
@modifications: Jakub Dotlacil, April 26, 2018
'''

import sys
import argparse
from datetime import datetime
# TODO: replace by the name of the package that you store these files in
# from naive_bayes import NaiveBayes
from os import listdir, remove, system
from os.path import isfile, join

# TODO: Please write the command that you use to call Python in the terminal here
# (most likely, the command is python or python3)
my_python = ""

def train_model(corpus_dir, classifier):
    '''Train a classifier on a training corpus where labels are provided.

    :param corpus_dir: The path to the training folder
    :param classifier: The classifier to be trained
    '''
    print('Starting training at {}'.format(datetime.now()))

    for directory in listdir(corpus_dir):
        print("Training on label {}".format(directory))
        directory_path = join(corpus_dir, directory)
        for text_file in listdir(directory_path):
            file_path = join(directory_path, text_file)
            classifier.update_label_count(directory)
            try:
                with open(file_path) as data_file:
                    classifier.train(data_file, directory)
            except IOError as e:
                print(e)
                print("It seems that the text_file {} is damaged.".format(text_file))
                sys.exit(0)

    print("Starting to smooth and normalise at {}".format(datetime.now()))
    classifier.smooth_feature_counts()
    classifier.log_normalise_label_probs()
    classifier.log_normalise_feature_probs()

    print("Finished training at {}".format(datetime.now()))

def make_predictions(predictions_file, test_dir, classifier):
    '''Make predictions on data with missing label

    :param predictions_file: The file to which the ouput predictions should be written
    :param test_dir: The path to the directory containing the test items
    :param classifier: A trained classifier
    '''
    print("Start making predictions at {}".format(datetime.now()))
    
    if isfile(predictions_file):
        remove(predictions_file)
    
    for test_file in listdir(test_dir):
        try:
            with open(join(test_dir, test_file)) as test, open(predictions_file, "a") as out:
                prediction = classifier.predict(test.read())
                out.write(test_file + "\t{}\n".format(prediction))
        except IOError as e:
            print(e)
            print("Something went wrong while reading test file {}".format(test_file))
            sys.exit(0)
            
    print("Finished making predictions at {}".format(datetime.now()))

def main():
    '''Standard method in Python that does not need a docstring. Don't worry about it for now, we will get to know it
    more deeply in week 6.
    '''
    
    corpus_dir = 'weekly_tasks/week4/homework/files_for_development/20news-18828'
    test_dir = 'weekly_tasks/week4/homework/files_for_testing/test-set'
    keys_file = 'weekly_tasks/week4/homework/files_for_testing/test_keys.txt'

    nb_classifier = NaiveBayes()

    if corpus_dir:
        train_model(corpus_dir, nb_classifier)

    if test_dir:
        make_predictions("predictions.txt", test_dir, nb_classifier)

    if keys_file:
        accuracy_checker(keys_file, "predictions.txt")
        # system("{} accuracy_checker.py {} predictions.txt".format(my_python, keys_file))


if __name__ == '__main__':
    main()

    accuracy_checker('predictions.txt', 'files_for_development/dev_keys.txt')