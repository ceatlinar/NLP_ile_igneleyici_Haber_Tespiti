
# Bu kernel iğneleyici haber başlıkları ile gerçek haber başlıklarını birbirinden ayırt etmek için LSTM'in kullanıldığı bir kernel örneğidir. Veri seti News Headlines Dataset for Sarcasm Detection veri setidir.

#Başlangıç kütüphanlerimizi import edelim
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

#Verimizi tanımak adına okuyup ilk satırlarını ekrana yazdıralım. Verimiz json dosyasında yer aldığı için pandas kütüphanesinin read_json metodundan faydalanacağız
df = pd.read_json("../input/Sarcasm_Headlines_Dataset.json", lines=True)
df.head(10)

#Gördüğümüz gibi verimiz üç öz nitelikten oluşmakta. Bunlardan article_link haber başlığının detaylı bilgisini içermekte olup işimize yaramayacaktır. headline özelliği haber başlığını ve is_sarcastic özelliği aynı zamanda verimizin sınıflarını belirten, haberin iğneleme haber olup olmadığını göstermektedir. Şimdi dataframe'den article_link özelliğini kullanmayacağımız için düşürelim *
df = df.drop('article_link', axis = 1)
df.head(7)

#Şimdi sınıf içi dağılımımızı görmek için verimizi görselleştirelim
import plotly as py
from plotly import graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
​
labels = ['Sarcastic', 'Non Sarcastic']
count_sarcastic = len(df[df['is_sarcastic']==1])
count_notsar = len(df[df['is_sarcastic']==0])
values = [count_sarcastic, count_notsar]
​
trace = go.Pie(labels=labels,
               values=values,
               textfont=dict(size=19, color='#FFFFFF'),
               marker=dict(
                   colors=['#080403', '#2424FF'] 
               )
              )
​
layout = go.Layout(title = '<b>Sarcastic vs Non Sarcastic</b>')
data = [trace]
fig = go.Figure(data=data, layout=layout)
​
iplot(fig)


#Haber başlıklarındaki en çok tekrar eden ilk 50 kelimeye bakalım
all_words = df['headline'].str.split(expand=True).unstack().value_counts()
data = [go.Bar(
            x = all_words.index.values[2:50],
            y = all_words.values[2:50],
            marker= dict(colorscale='Viridis',
                         color = all_words.values[2:100]
                        ),
            text='Word counts'
    )]
​
layout = go.Layout(
    title='Most Frequent Words in News Headlines'
)
​
fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='basic-bar')

#Yukarıdaki grafikten ilk 50 kelime içerisinde the, in, for gibi tek başlarına anlam ifade etmeyen ve stop words olarak adlandırılan yardımcı kelimelerin sıklıkla geçtiğini görmekteyiz. Bu kelimeler bize cümlenin ait olduğu sınıfı bulmamızda yardımcı olmayacağı için bunlardan kurtulmamız gerek.

#Stop wordslerden kurtulmadan önce haber başlıklarımızdan sayı ve noktalama işaretlerini, verimizi daha temiz hale getirmek ve başarıyı arttırmak adına çıkaralım

import string
from string import digits, punctuation
​
print("Before Punctuation and Digits Removing")
print(df['headline'][0])
​
headlines = []
for hl in df['headline']:
#   Noktalama işaretlerini çıkaralım
    clean = hl.translate(str.maketrans('', '', punctuation))
#   Sayıları çıkaralım
    clean = clean.translate(str.maketrans('', '', digits))
    headlines.append(clean)
print("After Punctuation and Digits Removing")
print(headlines[0])


#Şimdi de haber başlıklarımızı anlamlı parçalara, 'token'lara bölelim
print('Before Tokenazation')
print(headlines[0],"\n")
headlines_tokens = []
for hl in headlines:
    headlines_tokens.append(hl.split())
print("After Tokenazation")
print(headlines_tokens[0])


#Sıra geldi stop wordleri temizlemeye
import nltk
​
stopwords = nltk.corpus.stopwords.words('english')
final_data = []
for sentence in headlines_tokens:
    my_sentence = [word for word in sentence if word.lower() not in stopwords]
    final_data.append(my_sentence)
print("Before Stop Word Removing")
print(headlines_tokens[0],"\n")
print("After Stop Word Removing")
print(final_data[0])


flat_list = [item for sublist in final_data for item in sublist]
from collections import Counter
cnt = Counter(flat_list)
​
my_df = pd.DataFrame(list(cnt.items()), columns = ['Words', 'Freq'])
my_df = my_df.sort_values(by=['Freq'], ascending=False)
my_df_50 = my_df.head(50)
​
data = [go.Bar(
            y=my_df_50['Freq'],
            x=my_df_50['Words'],
            marker= dict(colorscale='Viridis',
                         color = all_words.values[2:100]
                        ),
            text='Word counts'
    )]
​
layout = go.Layout(
    title='Frequent Occuring word (unclean) in Headlines'
)
​
fig = go.Figure(data=data, layout=layout)
​
iplot(fig, filename='basic-bar')

#Yukarıdaki hücreyi çalıştırdığımızda gördüğümüz örnekte aynı iki cümlenin stop wordler çıkarılmadan önceki ve sonraki hali görünmektedir. Stop wordleri çıkardıktan sonra sırada lemmatization işlemi var. Bu işlem aynı kelimenin farklı çekimlerini kelimenin asıl hali ile değiştirme, asıl haline döndürme işlemidir.
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
​
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
​
    return tag_dict.get(tag, wordnet.NOUN)
​
lemmatizer = WordNetLemmatizer()
​
hl_lemmatized = []
for tokens in headlines_tokens:
    lemm = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    hl_lemmatized.append(lemm)
    
# Örnek Kıyaslama
word_1 = ['skyrim','dragons', 'are', 'having', 'parties']
word_2 = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_1]
print('Before lemmatization :\t',word_1)
print('After lemmatization :\t',word_2)
​

#Yukarıdaki kod parçacığı çalıştırıldığı zaman Lemmatization işleminin bir örnek çıktısı görülmektedir. Bir sonraki işlemimiz verimizi modelimize uygun hale getirmek
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np
​
max_features = 2000
max_token = len(max(hl_lemmatized))
tokenizer = Tokenizer(num_words=max_features);
tokenizer.fit_on_texts(hl_lemmatized)
sequences = tokenizer.texts_to_sequences(hl_lemmatized)
X = pad_sequences(sequences, maxlen=max_token)
index = 10
print('Önce :')
print(hl_lemmatized[index],'\n')
print('Sequences convertion sonrası:')
print(sequences[index],'\n')
print('Padding sonrası :')
print(X[index])

#Verimizi eğitim ve test için bölelim
from sklearn.model_selection import train_test_split
​
Y = df['is_sarcastic'].values
Y = np.vstack(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, random_state = 42)


#Şimdi de modelimizi oluşturalım
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
​
embed_dim = 64
​
model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = max_token))
model.add(LSTM(96, dropout=0.2, recurrent_dropout=0.2, activation='relu'))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
​
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


#Her şey hazır olduğuna göre sırada heyecanlı kısım var ; eğitim
epoch = 10
batch_size = 128
model.fit(X_train, Y_train, epochs = epoch, batch_size=batch_size, verbose = 2)


#Eğittiğimiz modeli test edelim
loss, acc = model.evaluate(X_test, Y_test, verbose=2)
print("Overall scores")
print("Accuracy\t: ", round(acc, 3))

#Şimdi de iki sınıfımız için de başarılarımızı görelim
pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_test)):
    
    result = model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.around(result) == np.around(Y_test[x]):
        if np.around(Y_test[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.around(Y_test[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1


print("Sarcasm accuracy\t: ", round(pos_correct/pos_cnt*100, 3),"%")
print("Non-sarcasm accuracy\t: ", round(neg_correct/neg_cnt*100, 3),"%")