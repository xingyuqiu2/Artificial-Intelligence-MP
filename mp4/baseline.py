# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    dic = {}
    dic_tag = {}
    for list in train:
        for pair in list:
            word, tag = pair
            if dic.get(word) == None:
                dic[word] = {tag:1}
            elif dic[word].get(tag) == None:
                dic[word][tag] = 1
            else:
                dic[word][tag] += 1
        
            if dic_tag.get(tag) == None:
                dic_tag[tag] = 1
            else:
                dic_tag[tag] += 1

    default_tag = max(dic_tag, key = dic_tag.get)
    dic_wt = {}
    for word in dic.keys():
        dic_wt[word] = max(dic[word], key = dic[word].get)

    res = []
    i = 0
    for list in test:
        res.append([])
        for word in list:
            if word not in dic_wt.keys():
                res[i].append((word, default_tag))
            else:
                res[i].append((word, dic_wt[word]))
        i += 1

    return res