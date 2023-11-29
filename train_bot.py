# Bibliotecas necessárias
import nltk
from nltk.stem import PorterStemmer
import json
import pickle
import numpy as np

# python -m nltk.downloader punkt

# Instância de um stemmer da NLTK para extrair o radical das palavras
stemmer = PorterStemmer()

# Inicialização de variáveis
words = []             # Lista para armazenar todas as palavras no arquivo de treinamento
classes = []           # Lista para armazenar todas as tags (categorias)
word_tags_list = []    # Lista para armazenar pares (palavras, tag) de cada padrão
ignore_words = ['?', '!', ',', '.', "'s", "'m"]  # Palavras a serem ignoradas durante o pré-processamento
train_data_file = open('intents.json').read()   # Carrega o arquivo JSON contendo padrões de treinamento
intents = json.loads(train_data_file)           # Analisa o arquivo JSON

def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words

#For para fazer a leitura dos dados do meu JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern_words = nltk.word_tokenize(pattern)
        words.extend(pattern_words)
        word_tags_list.append((pattern_words, intent['tag']))
    #Adiciona as tags a lista de classes
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

stem_words = get_stem_words(words, ignore_words)

print(stem_words)
print(word_tags_list[0])
print(classes)

def create_bot_corpus(stem_words, classes):
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))
    pickle.dump(stem_words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))
    return stem_words, classes

stem_words, classes = create_bot_corpus(stem_words, classes)
print(stem_words)
print(classes)

training_data = []
number_of_tags = len(classes)
labels = [0] * number_of_tags

#Criação dos sacos de palavras
for word_tags in word_tags_list:
    bag_of_words = []
    pattern_words = word_tags[0]

    for index, word in enumerate(pattern_words):
        pattern_words[index] = word
    
    #Cria o saco de palavras
    for word in stem_words:
        if word in pattern_words:
            bag_of_words.append(1)
        else: 
           bag_of_words.append(0) 
    print(bag_of_words)

    labels_encoding = list(labels)
    tag = word_tags[1]
    tag_index = classes.index(tag)
    labels_encoding[tag_index] = 1
    training_data.append([bag_of_words, labels_encoding])
print(training_data[0])

#Cria os dados de treinamento
def preprocess_train_data(training_data):
    training_data = np.array(training_data, dtype=object)
    train_x = list(training_data[:, 0])
    train_y = list(training_data[:, 1])

    print(train_x[0])
    print(train_y[0])
    return train_x, train_y

train_x, train_y = preprocess_train_data(training_data)
