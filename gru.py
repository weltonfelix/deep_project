import warnings

import datasets
import matplotlib.pyplot as plt
import math
import scipy.stats as ss

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pandas as pd
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from d2l import torch as d2l

import re
import nltk
import spacy
import emoji
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
warnings.filterwarnings('ignore', category=RuntimeWarning) # ignora avisos de runtime para evitar poluição do output
pd.set_option('display.max_columns', None) # print all columns

"""## Carregando o dataset"""

dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')
df = dataset['train'].to_pandas()

"""## Mudando valores booleanos para valores numéricos (0 e 1)"""

df = df.replace({True: 1.0, False: 0.0})

"""##### Como existe alguns valores NAN nessas colunas, é necessário realizar alguma modificação nos dados a fim de excluir esses valores. Como foram poucos, é possível simplesmente retirar as linhas que possuem coluna(s) com esse valor, com uma redução insignificante, menos de 0.2% do dataset

## Analisando esses valores NAN
"""

nan_rows = df[df.isna().any(axis=1)]

"""##### A remoção desses valores realmente não terá um impacto negativo no dataset. Logo, é possível realizar a retirada deles tranquilamente

## Excluindo colunas com valores NAN
"""

df.dropna(subset=["annotator_educ", "annotator_income", "annotator_ideology", "annotator_age"], inplace=True)

"""## Retirando as duas primeiras colunas (inúteis)"""

df.drop(["comment_id", "annotator_id"], axis=1, inplace=True)

"""##### Elas importavam apenas para a identificação de quem estava escrevendo os textos, que além de serem números únicos, sem repetição, não iriam influenciar o treinamento dos modelos

## Análise do balanceamento entre as classes das colunas categóricas
"""

"""## Detecção de outliers e remoção deles dependendo da quantidade"""

def calcular_limites_outliers(data):
    Q1 = data.quantile(0.25)  # primeiro quartil (25%)
    Q3 = data.quantile(0.75)  # terceiro quartil (75%)
    IQR = Q3 - Q1  # intervalo interquartil (IQR)
    limite_inferior = Q1 - 1.5 * IQR  # limite inferior para outliers
    limite_superior = Q3 + 1.5 * IQR  # limite superior para outliers
    return limite_inferior, limite_superior


for column in ["infitms", "outfitms", "annotator_severity", "std_err", "annotator_infitms", "annotator_outfitms", "hypothesis", "hate_speech_score"]:
    limite_inferior, limite_superior = calcular_limites_outliers(df[column]) # limites inferior e superior de outliers para a coluna atual
    outliers = df[(df[column] < limite_inferior) | (df[column] > limite_superior)] # identifica as linhas do dataset que são outliers
    

    if outliers.shape[0] < 1000: # threshold
        df = df[(df[column] >= limite_inferior) & (df[column] <= limite_superior)] # remove os outliers do dataset

def remove_special(text):
    for char in text: #Removendo caracteres especiais
      if ord(char.upper()) not in range(65, 91) and ord(char) != 32:
        text = text.replace(char, "")
    return ''.join(char for char in text if char not in emoji.EMOJI_DATA) #Removendo emojis

df['clean_text'] = df['text'].apply(remove_special)


"""## Removendo as stopwords"""

#Converter a coluna em um array
textos = df['text'].values

def remove_stopwords(text):
      stop_words = set(stopwords.words('english'))
      words = text.split()
      filtered_words = [word for word in words if word.lower() not in stop_words]
      return ' '.join(filtered_words)

df['no_stopwords'] = df['clean_text'].apply(remove_stopwords)

"""###Criando categorias para a coluna hate_speech_score"""

bins = [-1000, -1, 1, 1000]
labels = ["Normal", "Neutral", "Hate Speech"]

df['hate_speech_score_binned'] = pd.cut(df['hate_speech_score'], bins=bins, labels=labels)

"""#### Codificação de label"""

# Codificar os rótulos
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['hate_speech_score_binned'])

"""#### Divisão dos dados"""

# Dividir o dataset em treinamento e teste
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

"""### Lematização"""

nlp = spacy.load('en_core_web_sm') # Usando spaCy em vez do nltk para aplicar a lematização por ser mais eficiente

def lemmatize_text(text):
    lemmatized_text = []
    for doc in nlp.pipe(text, batch_size=50, n_process=-1): # Usar a biblioteca spacy aplicar processamento paralelo reduziu o tempo de execução dessa parte pela metade
        lemmatized_text.append(' '.join([token.lemma_ for token in doc]))
    return lemmatized_text

train_df['lemmatized_text'] = lemmatize_text(train_df['no_stopwords'])
test_df['lemmatized_text'] = lemmatize_text(test_df['no_stopwords'])

"""#### Tokenização do texto"""

# Função para tokenizar o texto
tokenizer = get_tokenizer("basic_english")

def tokenize(text):
    return tokenizer(text)

# Aplicar a tokenização ao dataframe
train_df['tokens'] = train_df['lemmatized_text'].apply(tokenize)
test_df['tokens'] = test_df['lemmatized_text'].apply(tokenize)

"""### Aplicando One-Hot Encoder, transformando tudo exceto os tokens em números"""

#Usando pandas get_dummies para One-Hot Encoding
hot_columns = []

for df in [train_df, test_df]:
  for column in df:
    if df[column].dtype == "bool":
      hot_columns.append(column)

train_hot = pd.get_dummies(train_df, columns=hot_columns)
test_hot = pd.get_dummies(test_df, columns=hot_columns)

for df in [train_df, test_df]:
  for column in df:
    if df[column].dtype == "bool" or df[column].dtype == "string":
      hot_columns.append(column)


for df in [train_hot, test_hot]:
  try: # tava dando problema se nao usasse isso aqui, mas acho que é porque eu tava no meio da execução, acho que nao precisa
    df.drop(['text', 'clean_text', 'no_stopwords', 'lemmatized_text', 'hate_speech_score_binned', 'annotator_gender',	'annotator_trans', 'annotator_educ', 'annotator_income', 'annotator_ideology'], axis=1, inplace=True)
  except:
    pass

"""#### Construção do vocabulário"""

# Construir vocabulário
def yield_tokens(data):
    for tokens in data:
        yield tokens

vocab = build_vocab_from_iterator(yield_tokens(train_hot['tokens']), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

"""### Transformando todos os números em um só tipo"""

for column in train_hot.columns:
  if column != "tokens":
    for df in [train_hot, test_hot]:
      df[column] = df[column].astype(np.float32)

#for column in train_hot:
  #print(train_hot[column].dtype)

"""### Transforma tokens em sequências numéricas que conseguem ser passadas pro modelo"""

for df in [train_hot, test_hot]:

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['tokens'])
    sequences = tokenizer.texts_to_sequences(df['tokens'])

    maxlen = 0
    for seq in sequences:
      if len(seq) > maxlen:
        maxlen = len(seq)

    padded_sequences = pad_sequences(sequences, maxlen=maxlen)

    padded_sequences = np.array(padded_sequences, dtype=np.float32)

    df['padded_sequences'] = list(padded_sequences)

"""#### Colocando as sequências como listas pra serem usadas pelos modelos"""

train_sequences = np.array(train_hot['padded_sequences'].tolist(), dtype=np.float32)
test_sequences = np.array(test_hot['padded_sequences'].tolist(), dtype=np.float32)

"""### Retirando colunas agora desnecessárias"""

for df in [train_hot, test_hot]:
  df.drop(["tokens"], axis=1, inplace=True)
  df.drop(["padded_sequences"], axis=1, inplace=True)

"""### Setando colunas que serão usadas pelo modelo pra prever"""

feature_columns = []

for column in train_hot.columns:
  if column not in ["label", "tokens", "padded_sequences"]:
    feature_columns.append(column)

"""### Criação dos datasets (GERAL PRA TODOS OS MODELOS)"""

class HateSpeechDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = self.labels[idx]
        return sequence, label

# Create updated datasets
train_dataset = HateSpeechDataset(train_sequences, train_hot['label'].values)
test_dataset = HateSpeechDataset(test_sequences, test_hot['label'].values)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor([int(label) for label in labels], dtype=torch.long)
    return sequences, labels

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

"""## Treinamento (GRU)

#### Função para calcular as métricas (GERAL PRA TODOS OS MODELOS)
"""

def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(labels, predicted)
    precision = precision_score(labels, predicted, average='weighted')
    f1 = f1_score(labels, predicted, average='weighted')
    auc_roc = roc_auc_score(labels, nn.functional.softmax(outputs, dim=1), multi_class='ovr')
    return accuracy, precision, f1, auc_roc

"""#### Criando a GRU"""

class GRUModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_classes=3):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.gru = nn.GRU(num_hiddens, num_hiddens, batch_first=True)
        self.fc = nn.Linear(num_hiddens, num_classes)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        output, _ = self.gru(embedded)
        output = output[:, -1, :]  # Use the output of the last time step
        output = self.fc(output)
        return output

# Instantiate the model

for otimizador in ["Adam", "SGD", "RMSprop"]:
   for lr in [0.0001, 0.001, 0.01]:
      for num_hiddens in [16, 32, 64]:
            print(f"Otimizador: {otimizador}, Learning Rate: {lr}, Número de Hidden Units: {num_hiddens}")
            
            num_inputs = len(vocab)
            gru_model = GRUModel(vocab_size=num_inputs, num_hiddens=32)
            criterion = nn.CrossEntropyLoss()
            
            if otimizador == "SGD":
               optimizer = optim.SGD(gru_model.parameters(), lr=lr, momentum=0.9)
            elif otimizador == "Adam":
                optimizer = optim.Adam(gru_model.parameters(), lr=lr)
            else:
                optimizer = optim.RMSprop(gru_model.parameters(), lr=lr, alpha=0.9)

            num_epochs = 10
            for epoch in range(num_epochs):
                gru_model.train()
                for inputs, labels in train_loader:
                    outputs = gru_model(inputs)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Evaluation on the test set
                gru_model.eval()
                all_labels = []
                all_outputs = []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        outputs = gru_model(inputs)
                        all_labels.extend(labels.cpu().numpy())
                        all_outputs.extend(outputs.cpu().numpy())

                all_labels = np.array(all_labels)
                all_outputs = np.array(all_outputs)
                accuracy, precision, f1, auc_roc = calculate_metrics(torch.tensor(all_outputs), torch.tensor(all_labels))

                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}, Precision: {precision}, F1-Score: {f1}, AUC-ROC: {auc_roc}')
