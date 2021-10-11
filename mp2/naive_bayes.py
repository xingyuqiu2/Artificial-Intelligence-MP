# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


import numpy as np
import string
def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    # return predicted labels of development set
    dic_ham = {}
    dic_spam = {}
    s = {}
    #train phase
    total_number_words_ham = 0
    total_number_words_spam = 0
    for i in range(len(train_set)):
        for word in train_set[i]:
            if s.get(word) == None:
                s[word] = 1
            else:
                s[word] += 1
            if train_labels[i] == 1:    #ham email
                total_number_words_ham += 1
                if dic_ham.get(word) == None:
                    dic_ham[word] = 1
                else:
                    dic_ham[word] += 1
            else:   #spam email
                total_number_words_spam += 1
                if dic_spam.get(word) == None:
                    dic_spam[word] = 1
                else:
                    dic_spam[word] += 1

    #development phase
    res = []

    for i in range(len(dev_set)):
        new_s = set()
        for word in dev_set[i]:
            if word not in new_s:
                new_s.add(word)
        new_distinct_words = len(new_s)

        p_ham = np.log(pos_prior)
        p_spam = np.log(1.0 - pos_prior)
        for word in dev_set[i]:
            p_ham += np.log(dic_ham.get(word, 0) + smoothing_parameter) - np.log(total_number_words_ham + (len(dic_ham) + new_distinct_words) * smoothing_parameter)
            
            p_spam += np.log(dic_spam.get(word, 0) + smoothing_parameter) - np.log(total_number_words_spam + (len(dic_spam) + new_distinct_words) * smoothing_parameter)
            
        if p_ham > p_spam:
            res.append(1)
        else:
            res.append(0)

    return res
    