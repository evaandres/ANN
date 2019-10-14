import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras.layers as layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Flatten
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from nltk.corpus import stopwords
from nltk import word_tokenize
from bs4 import BeautifulSoup
import cufflinks
from IPython.core.interactiveshell import InteractiveShell
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import re
import nltk
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk import word_tokenize
from keras.callbacks import *
import keras.callbacks as cb


import warnings
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from keras.utils.vis_utils import plot_model




#Import data

df = pd.read_csv('quejas.csv',sep="#",low_memory=False)
df = df[pd.notnull(df['Consumer complaint narrative'])]

print (df.info())
df.Product.value_counts()

print("input shape (rows,colums)",df.shape)


#Label Consolidation

df.loc[df['Product'] == 'Credit reporting', 'Product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
df.loc[df['Product'] == 'Credit card', 'Product'] = 'Credit card or prepaid card'
df.loc[df['Product'] == 'Payday loan', 'Product'] = 'Payday loan, title loan, or personal loan'
df.loc[df['Product'] == 'Virtual currency', 'Product'] = 'Money transfer, virtual currency, or money service'
df = df[df.Product != 'Other financial service']

cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


fig = plt.figure(figsize=(10,10))
df['Product'].value_counts().sort_values(ascending=False).plot.bar(ylim=0)
plt.show()


#Text Preprosessing

df = df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z +_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text


df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(clean_text)
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].str.replace('\d+', '')



#vectorize consumer complaints text, by turning each text into either a sequence of integers or into a vector.
#Limit the data set to the top 200 words.
#Set the max number of words in each complaint at 100.


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 2000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 2000
# This is fixed.    
EMBEDDING_DIM = 200
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Consumer complaint narrative'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#Truncate and pad the input sequences so that they are all in the same length for modeling.

X = tokenizer.texts_to_sequences(df['Consumer complaint narrative'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

#Converting categorical labels(products) to numbers.
Y = pd.get_dummies(df['Product']).values
print('Shape of label tensor:', Y.shape)

#Traing and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 42)
print("(X,Y) train.shape",X_train.shape,Y_train.shape)
print("(X,Y) test shape",X_test.shape,Y_test.shape)

#Model

model = Sequential()
#Equivalent input layer tokenizing the input keeping into account vocab size,epoch to training this codification entry data ,output vector size(max size of a complaint)
#This layer will be trained separately of the neural network and this could be use for other neural networks, as part o neural network o for pre-trainnee
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1])) 
lstm1 = model.add(LSTM(50,activation='tanh', recurrent_activation='hard_sigmoid', dropout=0.25, recurrent_dropout=0.25, return_sequences=True)) 
#activation=hiperbolyc tangent. dropout ignoring only 10% of the units (output)
model.add(LSTM(20,activation='tanh',recurrent_activation='hard_sigmoid', dropout=0.45, recurrent_dropout=0.45))
model.add(Dense(9, activation='softmax'))
#Dense. with this I am using densely connected nn (each neuron of a layer is connected to all the neurons of the next layers) instead of convolutional 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
plot_model(model,'Lstm-demo5.png', show_shapes=True)




epochs = 5 #number of iterations over the full set of samples
batch_size = 16 #number of samples before each training of the nn (number of complains) 32-overfitting 16-pico of of 10 a bit of

tensorboard = cb.TensorBoard(log_dir='./logs', histogram_freq=50, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[tensorboard])

#evaluation

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

plt.title('Loss')
plt.plot(history.history['loss'], label='train') #trainning loss
plt.plot(history.history['val_loss'], label='test') #validation loss or test loss
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()

#Test with a New Complaint

new_complaint = ["We opened our first mortgage on a house with the loan servicing company Amerifirst. The loan amount was approx. {$170000.00} and we closed on the house XX/XX/2019"]
text1 =  ["This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."]
text2 =  ["I loss my credit card for that I need a new one"] 
text3 =  ["I am disputing with my student loan car paypal"]
text4 =  ["Freedom mortgage company 's insurance department made an error at the end of XXXX but writing a check to the wrong homeowners insurance company. That check was then cancelled but the money wasn't refunded to the escrow account for over 5 months."]
seq = tokenizer.texts_to_sequences(new_complaint)
seq1 = tokenizer.texts_to_sequences(text1)
seq2 = tokenizer.texts_to_sequences(text2)
seq3 = tokenizer.texts_to_sequences(text3)
seq4 = tokenizer.texts_to_sequences(text4)


padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
padded1 = pad_sequences(seq1, maxlen=MAX_SEQUENCE_LENGTH)
padded2 = pad_sequences(seq2, maxlen=MAX_SEQUENCE_LENGTH)
padded3 = pad_sequences(seq3, maxlen=MAX_SEQUENCE_LENGTH)
padded4 = pad_sequences(seq4, maxlen=MAX_SEQUENCE_LENGTH)


pred = model.predict(padded)
pred1 = model.predict(padded1)
pred2 = model.predict(padded2)
pred3 = model.predict(padded3)
pred4 = model.predict(padded4)



labels = ['Credit reporting, credit repair services, or other personal consumer reports', 'Debt collection', 'Mortgage', 'Credit card or prepaid card', 'Student loan', 'Bank account or service', 'Checking or savings account', 'Consumer Loan', 'Payday loan, title loan, or personal loan', 'Vehicle loan or lease', 'Money transfer, virtual currency, or money service', 'Money transfers', 'Prepaid card']
print(new_complaint,labels[np.argmax(pred)])
print(text1,labels[np.argmax(pred1)])
print(text2,labels[np.argmax(pred2)])
print(text3,labels[np.argmax(pred3)])
print(text4,labels[np.argmax(pred4)])

model.save("modeldemo5.h5")












