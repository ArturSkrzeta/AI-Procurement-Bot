import json
import random
import numpy as np
import tensorflow as tf
import tflearn
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle

#Global
stemmer = LancasterStemmer()

def build_model(first_layer,last_layer):

    tf.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(first_layer[0])])
    # 2 hidden layers
    net = tflearn.fully_connected(net,8)
    net = tflearn.fully_connected(net,8)
    net = tflearn.fully_connected(net,len(last_layer[0]), activation="softmax") # softmax assesses probability # ending layer = outputs
    # we take the greates predicted label, we grab the resposes from it,  we radomnly pick one repsonse and give back to user
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    # try:
    #     model.load('model.tflearn')
    # except:
    model.fit(first_layer, last_layer, n_epoch=1000, batch_size=8, show_metric=True) # n_epoch - how many times it sees the data
    model.save('model.tflearn')

    return model

def bag_of_words_from_user(sentence, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s_word in s_words:
        for i,w in enumerate(words):
            if w == s_word:
                bag[i] = 1

    return np.array(bag)

def chat(data, model,words, labels):
    print("Start talking with the bot! - type quit to stop")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words_from_user(inp,words)])[0] # predict method accepts a list -- [[]] -- list in a list
        results_index = np.argmax(results)             # returns probabilities of each labels - we need to take the highest one
        tag = labels[results_index]

        if results[results_index] > 0.5:
            for tg in data['category']:
                if tg['tag'] == tag:
                    responses = tg['responses']

            return random.choice(responses)
        else:
            return "I didn't get that, try again in some other way."


def bag_of_words_from_bot():

    with open('data.json') as file:
        data = json.load(file)

    try:
        # data.pickle
        with open('', 'rb') as f:
            words, labels, training, output = pickle.load(f)
    except:

        words = []
        labels = []
        docs_x = []
        docs_y= []

        for item in data['category']:
            for pattern in item['patterns']:
                wrds = nltk.word_tokenize(pattern) # one word = one token >> list of tokens
                words.extend(wrds) # extending words list with tokens

                # each token has responding tag
                docs_x.append(wrds)
                docs_y.append(item['tag'])

            if item['tag'] not in labels:
                labels.append(item['tag'])

        #print(words)       # ['Hi','How','are','you','Is',...]
        #print(docs_x)      # [['Hi'],['How','are','you'],...}
        #print(docs_y)      # ['greeting','greeting',...]
        words = [stemmer.stem(w.lower()) for w in words if w != "?"] # taking words to the root
        #print(words)       # ['hi','how','ar'.'you'.'is','anyon','ther',...]
        words = sorted(list(set(words))) # removing duplicates and sorting
        #print(words)        # ['a','ag','am','anyon',...]
        labels = sorted(labels)

        #### CONVERTING STRINGS TO NUMBERS #####

        training = []
        output = []
        out_empty = [0 for _ in range(len(labels))] # as many zeros as amount of labels

        for x, doc in enumerate(docs_x): #each pattern in patterns
            bag = []
            #print(doc)
            wrds = [stemmer.stem(w) for w in doc] # stemming pattern -> breaking patterns down to letters
                                                    # not tokenized
            #print(wrds) # ['hi']
                        # ['how','ar','you']
                        # ['is', 'anyon', 'ther', '?']...

            #if token in tokens is in pattern - list of words that make a pattern
            for w in words: # for each word in ['a','ag','am','anyon',...]
                if w in wrds: # if 'a' in ['how','ar','you'] >>> bag.append(1)
                    bag.append(1) # if token w exists in a pattern wrds list
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1 # when label hit then 1 for every pattern

            #print(output_row)    # [0,0,1,0,0,0]
                                # [0,0,1,0,0,0]
            training.append(bag)
            output.append(output_row)

        # for list in training:
        #     print(list)

        # for list in output:
        #     print(list)

        training = np.array(training) # = count of patterns (26) x count of words (46)
        output = np.array(output) # = count of patterns (26) x count of lables (6)

        with open('data.pickle', 'wb') as f:
            pickle.dump((words, labels, training, output),f)

    return data, training, output, words, labels


# data, training, output, words, labels = bag_of_words_from_bot()
# model = build_model(training,output)
# chat(data, model, words, labels)
