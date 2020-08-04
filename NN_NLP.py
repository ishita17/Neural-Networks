#classifying tweets
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

nltk.download('words')
nltk.download('punkt')
stop = stopwords.words('english')
nltk.download('words')
words = set(nltk.corpus.words.words())
nltk.download('wordnet')

#loading data
train = pd.read_csv(r'C:\Users\ishit\OneDrive\Desktop\New Hard Drive\Portfolio\Project1\train.csv')
test = pd.read_csv(r'C:\Users\ishit\OneDrive\Desktop\New Hard Drive\Portfolio\Project1\test.csv')

#shuffling data
train = shuffle(train)
x_train = train['text']
x_test = test['text']
y_train = train['target']

#data cleaning
stop_words = set(stopwords.words('english'))

def data_clean(corpus):
    corpus = corpus.map(lambda s: s.lower() if type(s) == str else s )
    corpus = corpus.map(lambda s: re.sub('\W', ' ', s))
    corpus = corpus.map(lambda x: word_tokenize(x))
    corpus = corpus.map(lambda s: [item for item in s if item not in stop])
    corpus = corpus.map(lambda s: [WordNetLemmatizer().lemmatize(i) for i in s])
    corpus = corpus.map(lambda s: [item for item in s if item in words])
    corpus = corpus.apply(lambda s: ' '.join(s))
    return corpus

xtrain = data_clean(x_train)
xtest = data_clean(x_test)

#bag of words count vectorization
count = CountVectorizer()
xtrainV = count.fit_transform(x_train).toarray()

feat_dict = sorted(count.vocabulary_.keys())
tokens = count.get_feature_names()
vectors = pd.DataFrame(data = xtrainV, columns = tokens)

#vectorizing test data set
xtestV = count.transform(xtest).toarray()

#model
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape=(21637,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid')) 

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(xtrainV, y_train, validation_split = 0.2,\
                    epochs = 15, batch_size = 2000 )

# this is the plot for the loss

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss'] 

epochs = range(1, 26)

plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#this is the plot for accuracy

plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, history.history['accuracy'], 'bo', label = 'Training acc')
plt.plot(epochs, history.history['val_accuracy'], 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#making test predictions
results = model.predict(xtestV)






















