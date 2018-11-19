"""
The file implementing Naive Bayes classifier.
"""
###
# You have to work on this file. Your task is to implement the methods
# marked as TODO.

from collections import Counter
import math

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
        for line in data:
            self.add_feature_counts(line.split(), label)
    
    def add_feature_counts(self, features, label):
        '''
        Count the features in a feature list.
        
        :param features: a list of features.
        :param label: the label of the data file from which the features were extracted.
        '''
        # This method updates feature_counts by features given the label. It
        # should also update vocabulary with features.
        # TODO: implement this!
        pass

    def smooth_feature_counts(self, smoothing=1):
        '''Smooth the collected feature counts

        :param smoothing: The smoothing constant
        '''
        # This method smoothes counts in feature_counts. Check the assignment
        # description on how to do this.
        # TODO: implement this!
        pass
        
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
        # TODO: Implement this!
        pass
            
    def log_normalise_feature_probs(self):
        '''
        Take feature counts in feature_counts and for each label, normalize
        them to probabilities and turn them into logprobs. update
        feature_probs with the created logprobs.
        '''
        # Take feature_counts, update feature_probs.
        # feature_probs have labels as keys. The values are
        # dictionaries that have features as keys as log-probs as values.
        # TODO: Implement this!
        pass
                
    def predict(self, data):
        ''' 
        Predict the most probable label according to the model on a stream of data.
        
        :param data: A stream of string data from which to extract features
        :return: the most probable label for the data (type string)
        '''
        # TODO: implement this!
        pass 
