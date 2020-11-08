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
data_source = 'data.json'

def bag_of_words_from_user(sentence, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s_word in s_words:
        for i,w in enumerate(words):
            if w == s_word:
                bag[i] = 1

    return np.array(bag)

def bag_of_words_from_bot():

    with open(sata_source) as file:
        data = json.load(file)

    try:
        # data.pickle when data processed last time
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

        # print(words)       # ['Hi','How','are','you','Is',...]        # single words - tokenized
        # pring(labels)      # ['greeting', 'goodbye','access'...]      # unique tags
        # print(docs_x)      # [['Hi'],['How','are','you'],...]
        # print(docs_y)      # ['greeting','greeting',...]              # tags corresponding with tokenized sentence
        words = [stemmer.stem(w.lower()) for w in words if w != "?"] # taking words to the root
        #print(words)       # ['hi','how','ar'.'you'.'is','anyon','ther',...]
        words = sorted(list(set(words))) # removing duplicates and sorting
        #print(words)        # ['a','ag','am','anyon',...]
        labels = sorted(labels)
        # print(labels)     # ['access','goodbye','greeting','password', 'purchase-it', 'purchase-office']

        #### CONVERTING STRINGS TO NUMBERS #####

        training = []
        output = []
        out_empty = [0 for _ in range(len(labels))] # as many zeros as amount of labels

        for x, doc in enumerate(docs_x): #each pattern in patterns
            bag = []
            wrds = [stemmer.stem(w) for w in doc]   # stemming pattern, not tokenized
            # print(wrds) [['hi'], ['how','ar','you'], ['is', 'anyon', 'ther', '?']]

            #if token in tokens is in pattern - list of words that make a pattern
            for w in words: # for each word in ['a','ag','am','anyon',...]
                if w in wrds: # if token 'a' in stemmed list of words ['hi']] >>> bag.append(1)
                    bag.append(1) # if token w exists in a pattern wrds list
                else:
                    bag.append(0)

            output_row = out_empty[:]               # shallow copy
            output_row[labels.index(docs_y[x])] = 1 # when label hit then 1 for every pattern

            #print(output_row)  # [0,0,1,0,0,0] --> label of index 2 --> 'greeting'

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


def build_model(first_layer,last_layer):

    tf.reset_default_graph()

    # BUILDING NEURAL NETWORK
    # first layer with data input
    net = tflearn.input_data(shape=[None, len(first_layer[0])])
    # 2 hidden layers
    net = tflearn.fully_connected(net,8)
    net = tflearn.fully_connected(net,8)
    # last layer wit activation function and data output
    net = tflearn.fully_connected(net,len(last_layer[0]), activation="softmax") # softmax assesses probability # ending layer = outputs
    # we take the greatest predicted label, we grab the resposes from it, we radomnly pick one repsonse and give back to user
    net = tflearn.regression(net)

    # pickinng Deep Neural Network as a model
    model = tflearn.DNN(net)

    # Try to load model when was built last time
    # try:
    #     model.load('model.tflearn')
    # except:
    model.fit(first_layer, last_layer, n_epoch=1000, batch_size=8, show_metric=True) # n_epoch - how many times it sees the data
    model.save('model.tflearn')

    return model

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



def main():

    data, training, output, words, labels = bag_of_words_from_bot()
    model = build_model(training,output)
    chat(data, model, words, labels)


if __name__ == "__main__":
    main()
