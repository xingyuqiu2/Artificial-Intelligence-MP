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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math

def get_info(train):
    tag_count = {}
    word_count = {}
    word_dic_tag_count = {}
    tag_dic_word_count = {}
    pretag_dic_tag_count = {}
    for list in train:
        pretag = "START"
        for word, tag in list[1:]:
            tag_count[tag] = tag_count.get(tag, 0) + 1
            word_count[word] = word_count.get(word, 0) + 1
            # get word -> {tag -> count}
            if word in word_dic_tag_count:
                temp_tag_count = word_dic_tag_count[word]
                temp_tag_count[tag] = temp_tag_count.get(tag, 0) + 1
                word_dic_tag_count[word] = temp_tag_count
            else:
                word_dic_tag_count[word] = {tag:1}
            
            # get tag -> {word -> count}
            if tag in tag_dic_word_count:
                temp_word_count = tag_dic_word_count[tag]
                temp_word_count[word] = temp_word_count.get(word, 0) + 1
                tag_dic_word_count[tag] = temp_word_count
            else:
                tag_dic_word_count[tag] = {word:1}

            # get pretag -> {tag -> count}
            if pretag in pretag_dic_tag_count:
                temp_tag_count = pretag_dic_tag_count[pretag]
                temp_tag_count[tag] = temp_tag_count.get(tag, 0) + 1
                pretag_dic_tag_count[pretag] = temp_tag_count
            else:
                pretag_dic_tag_count[pretag] = {tag:1}
            pretag = tag
    
    return word_dic_tag_count, tag_dic_word_count, pretag_dic_tag_count, tag_count, word_count



def convert_to_prob(tag_dic_sth_count, k):
    # P(A|B) = ( count(A,B)+k ) / ( count(B)+k∗|unique types in A+1| )
    B_dic_A_prob = {}
    for tag, dic in tag_dic_sth_count.items():
        num_unique_types = len(dic)
        count_B = sum(dic.values())
        denominator = count_B + k * (num_unique_types + 1)
        dic_A_prob = {}
        for A, count_A_B in dic.items():
            numerator = count_A_B + k
            dic_A_prob[A] = math.log(numerator) - math.log(denominator)
        dic_A_prob["UNK"] = math.log(k) - math.log(denominator)
        B_dic_A_prob[tag] = dic_A_prob
    return B_dic_A_prob
        

def backtrack(sentence, pretag_prob, word_dic_tag_count, emission_porb, transition_prob, tag_count, prob_unknown_word):
    # P(cur_tag_k) = max( mu(pre_tag_i) * P(cur_tag | pre_tag_i) * P(cur_word | cur_tag) )
    # cur_tag_k -> argmax( P(cur_tag_k) )
    # return pretag_chosen, path
    word = sentence[0]
    tag_prob = {}
    dic_curtag_pretag = {}
    if word not in word_dic_tag_count:
        for cur_tag in tag_count.keys():
            # max probility for cur_tag
            P_cur_tag = -99999999
            for pre_tag, P_pre in pretag_prob.items():
                if pre_tag not in transition_prob:
                    transition_prob[pre_tag] = {"UNK":math.log(1e-5) - math.log(tag_count[pre_tag])}
                if cur_tag not in transition_prob[pre_tag]:
                    transition_prob[pre_tag][cur_tag] = transition_prob[pre_tag]["UNK"]
                    
                prob = P_pre + transition_prob[pre_tag][cur_tag] + prob_unknown_word
                if prob > P_cur_tag:
                    P_cur_tag = prob
                    # point to pre_tag for path finding
                    dic_curtag_pretag[cur_tag] = pre_tag
            tag_prob[cur_tag] = P_cur_tag
    else:
        for cur_tag in word_dic_tag_count[word].keys():
            # max probility for cur_tag
            P_cur_tag = -99999999
            for pre_tag, P_pre in pretag_prob.items():
                if pre_tag not in transition_prob:
                    transition_prob[pre_tag] = {"UNK":math.log(1e-5) - math.log(tag_count[pre_tag])}
                if cur_tag not in transition_prob[pre_tag]:
                    transition_prob[pre_tag][cur_tag] = transition_prob[pre_tag]["UNK"]
                if word not in emission_porb[cur_tag]:
                    emission_porb[cur_tag][word] = prob_unknown_word
                    
                prob = P_pre + transition_prob[pre_tag][cur_tag] + emission_porb[cur_tag][word]
                if prob > P_cur_tag:
                    P_cur_tag = prob
                    # point to pre_tag for path finding
                    dic_curtag_pretag[cur_tag] = pre_tag
            tag_prob[cur_tag] = P_cur_tag
            
    if len(sentence) == 1:
        tag_chosen = max(tag_prob, key = tag_prob.get)
        return dic_curtag_pretag[tag_chosen], [(word, tag_chosen)]
    tag_chosen, path = backtrack(sentence[1:], tag_prob, word_dic_tag_count, emission_porb, transition_prob, tag_count, prob_unknown_word)
    return dic_curtag_pretag[tag_chosen], [(word, tag_chosen)] + path
        
                               

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    word_dic_tag_count, tag_dic_word_count, pretag_dic_tag_count, tag_count, word_count = get_info(train)
    emission_porb = convert_to_prob(tag_dic_word_count, 1e-5)
    transition_prob = convert_to_prob(pretag_dic_tag_count, 1e-5)
    # find default value for unseen word
    res = []
    set_newWords = set()
    for sentence in test:
        for word in sentence:
            if word not in word_count and word not in set_newWords:
                set_newWords.add(word)
    prob_unknown_word = math.log(1e-5) - math.log(sum(word_count.values()) + 1e-5 * (len(word_count) + len(set_newWords)))
    # use viterbi algorithm
    for sentence in test:
        tag_prob = {"START":math.log(1)}
        path = backtrack(sentence[1:], tag_prob, word_dic_tag_count, emission_porb, transition_prob, tag_count, prob_unknown_word)[1]
        path = [(sentence[0], "START")] + path
        res.append(path)
    return res