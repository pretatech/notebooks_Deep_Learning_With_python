"""
Geração de texto LSTM com base na incorporação de palavras
"""

import random
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow import keras
import numpy as np
import jieba
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

path = '/Users/c/Desktop/b.txt'
text = open(path).read().lower().replace('\n', '').replace('　　', '\n')
print('Corpus length:', len(text))

# Vetorizar sequência de texto

maxlen = 60     # Comprimento de cada sequência
step = 3        # Experimente uma nova sequência a cada 3 tokens
sentences = []  # Salve a sequência extraída
next_tokens = []  # frases próximo token

print('Vectorization...')

token_text = list(jieba.cut(text))

tokens = list(set(token_text))
tokens_indices = {token: tokens.index(token) for token in tokens}
print('Number of tokens:', len(tokens))

for i in range(0, len(token_text) - maxlen, step):
    sentences.append(
        list(map(lambda t: tokens_indices[t], token_text[i: i+maxlen])))
    next_tokens.append(tokens_indices[token_text[i+maxlen]])
print('Number of sequences:', len(sentences))

next_tokens_one_hot = []
for i in next_tokens:
    y = np.zeros((len(tokens),), dtype=np.bool)
    y[i] = 1
    next_tokens_one_hot.append(y)

dataset = tf.data.Dataset.from_tensor_slices((sentences, next_tokens_one_hot))
dataset = dataset.shuffle(buffer_size=4096)
dataset = dataset.batch(128)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# Modelo de construção

model = models.Sequential([
    layers.Embedding(len(tokens), 256),
    layers.LSTM(256),
    layers.Dense(len(tokens), activation='softmax')
])

# Configuração de compilação de modelo

optimizer = optimizers.RMSprop(lr=0.1)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer)

# Função de amostragem


def sample(preds, temperature=1.0):
    '''
    Pesar novamente a distribuição de probabilidade original obtida pelo modelo e extrair um índice de token dele
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Modelo de treinamento


callbacks_list = [
    # Economize pesos após cada rodada
    keras.callbacks.ModelCheckpoint(
        filepath='text_gen.h5',  # Caminho para salvar o arquivo
        monitor='loss',      # monitor：Indicador a ser verificado
        save_best_only=True,     # Apenas salve o modelo com o melhor indicador de monitor (se o monitor não melhorar, ele não será salvo)
    ),
    # Reduza a taxa de aprendizagem quando não estiver mais melhorando
    keras.callbacks.ReduceLROnPlateau(
        monitor='loss',    # Indicador a ser verificado
        factor=0.5,            # Quando acionado: taxa de aprendizagem * = fator
        patience=1,            # Se o monitor não melhorar na rodada de paciência, ele aciona para reduzir a taxa de aprendizagem
    ),
    # Interrompa o treinamento quando não estiver mais melhorando
    keras.callbacks.EarlyStopping(
        monitor='loss',           # Indicador a ser verificado
        patience=3,             # Se o monitor não melhorar em mais do que rodadas de paciência (por exemplo, 10 + 1 = 11 rodadas aqui), pare o treinamento
    ),
]

model.fit(dataset, epochs=60, callbacks=callbacks_list)

# Geração de texto

start_index = random.randint(0, len(text) - maxlen - 1)
generated_text = text[start_index: start_index + maxlen]
# print(f' Generating with seed: "{generated_text}"')
print(f'  📖 Generating with seed: "\033[1;32;43m{generated_text}\033[0m"')

for temperature in [0.2, 0.5, 1.0, 1.2]:
    # print('\n  temperature:', temperature)
    print(f'\n   \033[1;36m 🌡️ temperature: {temperature}\033[0m')
    print(generated_text, end='')
    for i in range(400):    # Gerar 400 tokens
        # Codifique o texto atual
        text_cut = jieba.cut(generated_text)
        sampled = []
        for i in text_cut:
            if i in tokens_indices:
                sampled.append(tokens_indices[i])
            else:
                sampled.append(0)

        # Prever, provar, gerar o próximo token
        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_token = tokens[next_index]
        print(next_token, end='')

        generated_text = generated_text[1:] + next_token
