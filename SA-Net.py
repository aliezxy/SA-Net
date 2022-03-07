import itertools
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def readFa(fa):
    with open(fa,'r') as FA:
        seqName,seq,struct_seq='','',''
        while 1:
            line=FA.readline()
            line=line.strip('\n')
            if (line.startswith('>') or not line) and seqName:
                yield((seqName,seq,struct_seq))
            if line.startswith('>'):
                seqName = line[1:]
                seq=''
                struct_seq=''
            elif line.startswith('(') or line.startswith('.'):
                struct_seq+=line
            else:
                seq+=line
            if not line:break

def init_seq_dictionary(kmer):
    ans = ''
    nucleotide_list = ['a', 'u', 'c', 'g']

    temp = [''.join(x) for x in itertools.product(*[nucleotide_list] * kmer)]

    for i in temp:
        ans = ans + i + ' '

    return ans

def seq_to_kmer(sentence, kmer):
    ans = ''

    for i in range(0,len(sentence)-kmer+1):
        ans = ans + sentence[i:i+kmer] + ' '

    return ans

def init_data(datasetname,kmer):
    training_seq_positives = []
    training_labels_positives = []

    fa="/"+ datasetname + ".train.positives.fa"
    for seqName,seq,struct_seq in readFa(fa):
      training_seq_positives.append(seq)
      training_labels_positives.append(1)

    training_seq_negatives = []
    training_labels_negatives = []

    fa="/"+ datasetname + ".train.negatives.fa"
    for seqName,seq,struct_seq in readFa(fa):
      training_seq_negatives.append(seq)
      training_labels_negatives.append(0)

    training_seq = []
    training_labels = []

    while len(training_seq_positives) != 0:
      training_seq.append(training_seq_positives.pop())
      training_labels.append(training_labels_positives.pop())
      if len(training_seq_negatives) != 0:
          training_seq.append(training_seq_negatives.pop())
          training_labels.append(training_labels_negatives.pop())
    while len(training_seq_negatives) != 0:
      training_seq.append(training_seq_negatives.pop())
      training_labels.append(training_labels_negatives.pop())

    testing_seq_positives = []
    testing_labels_positives = []

    fa="/"+ datasetname + ".ls.positives.fa"
    for seqName,seq,struct_seq in readFa(fa):
      testing_seq_positives.append(seq)
      testing_labels_positives.append(1)

    testing_seq_negatives = []
    testing_labels_negatives = []

    fa="/"+ datasetname + ".ls.negatives.fa"
    for seqName,seq,struct_seq in readFa(fa):
      testing_seq_negatives.append(seq)
      testing_labels_negatives.append(0)

    testing_seq = []
    testing_labels = []

    while len(testing_seq_positives) != 0:
      testing_seq.append(testing_seq_positives.pop())
      testing_labels.append(testing_labels_positives.pop())
      if len(testing_seq_negatives) != 0:
          testing_seq.append(testing_seq_negatives.pop())
          testing_labels.append(testing_labels_negatives.pop())

    while len(testing_seq_negatives) != 0:
      testing_seq.append(testing_seq_negatives.pop())
      testing_labels.append(testing_labels_negatives.pop())

    training_sequence = []

    for sentence in training_seq:
      temp = seq_to_kmer(sentence, kmer)
      training_sequence.append(temp)

    testing_sequence = []

    for sentence in testing_seq:
      temp = seq_to_kmer(sentence, kmer)
      testing_sequence.append(temp)

    tokenizer_seq = Tokenizer()
    tokenizer_seq.fit_on_texts([init_seq_dictionary(kmer)])

    max_length_seq = len(training_seq[0])-kmer+1
    trunc_type='post'
    padding_type='post'

    training_sequences = tokenizer_seq.texts_to_sequences(training_sequence)
    training_seq_padded = pad_sequences(training_sequences, maxlen = max_length_seq, padding = padding_type, truncating = trunc_type)

    testing_sequences = tokenizer_seq.texts_to_sequences(testing_sequence)
    testing_seq_padded = pad_sequences(testing_sequences, maxlen = max_length_seq, padding = padding_type, truncating = trunc_type)

    training_seq_padded = np.array(training_seq_padded)
    training_labels = np.array(training_labels)

    testing_seq_padded = np.array(testing_seq_padded)
    testing_labels = np.array(testing_labels)

    return training_seq_padded, training_labels, testing_seq_padded, testing_labels

def auroc(y_true, y_pred):
    return tf.compat.v1.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, input):
        batch_size = tf.shape(input)[0]

        q = self.wq(input)
        k = self.wk(input)
        v = self.wv(input)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,(batch_size, -1, d_model))
        output = self.dense(concat_attention)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
      return tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),
          tf.keras.layers.Dense(d_model)
      ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x):
        attn_output, _ = self.mha(x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, input_vocab_size):
        super(EmbeddingLayer, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

    def call(self, x):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        return x

class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate):
        super(SelfAttentionLayer, self).__init__()
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for i in range(num_layers):
          x = self.enc_layers[i](x)
        return x

class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, rate):
        super(OutputLayer, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x


class SA_Net(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.05):
        super(SA_Net, self).__init__()
        self.embeddingLayer = EmbeddingLayer(d_model, input_vocab_size)
        self.selfAttentionLayer = SelfAttentionLayer(num_layers, d_model, num_heads, dff, maximum_position_encoding, rate)
        self.outputLayer = OutputLayer(rate)

    def call(self, x):
        x = self.embeddingLayer(x)
        x = self.selfAttentionLayer(x)
        x = self.outputLayer(x)
        return x

kmer = 4
num_layers = 6
num_heads = 2
rate = 0.05
d_model = 10
dff = d_model / 2

datasetnames = ["ALKBH5_Baltz2012","C17ORF85_Baltz2012","C22ORF28_Baltz2012","CAPRIN1_Baltz2012","CLIPSEQ_AGO2",
"CLIPSEQ_ELAVL1","CLIPSEQ_SFRS1","ICLIP_HNRNPC","ICLIP_TDP43","ICLIP_TIA1","ICLIP_TIAL1",
"PARCLIP_AGO1234","PARCLIP_ELAVL1","PARCLIP_ELAVL1A","PARCLIP_EWSR1","PARCLIP_FUS","PARCLIP_HUR",
"PARCLIP_IGF2BP123","PARCLIP_MOV10_Sievers","PARCLIP_PUM2","PARCLIP_QKI","PARCLIP_TAF15","PTBv1","ZC3H7B_Baltz2012"]

# for datasetname in datasetnames:
#     training_seq, training_labels, testing_seq, testing_labels = init_data(datasetname=datasetname, kmer=kmer)
#     input = tf.keras.Input((len(training_seq[0])))
#     prediction = SA_Net(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
#                         input_vocab_size=4 ** kmer + 1, maximum_position_encoding=len(training_seq[0]), rate=rate)(input)
#
#     model = tf.keras.models.Model(inputs=input, outputs=prediction)
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', auroc])
#
#     checkpoint_path = "/checkpoints/" + datasetname
#     ckpt = tf.train.Checkpoint(model=model)
#     ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
#
#     if ckpt_manager.latest_checkpoint:
#         ckpt.restore(ckpt_manager.latest_checkpoint)
#         print('Latest checkpoint restored!!')
#
#     num_epochs = 100
#     mini_batches = 128
#     earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_auroc', mode='max', min_delta = 0, patience = 3, verbose = 1)
#     history = model.fit(training_seq, training_labels, batch_size=mini_batches, epochs=num_epochs, validation_data=(testing_seq_padded, testing_labels), callbacks = earlystop,verbose=1)
#     ckpt_save_path = ckpt_manager.save()
#     print ('Saving checkpoint at {}'.format(ckpt_save_path))

for datasetname in datasetnames:
    training_seq, training_labels, testing_seq, testing_labels = init_data(datasetname=datasetname, kmer=kmer)
    input = tf.keras.Input((len(training_seq[0])))
    prediction = SA_Net(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                        input_vocab_size=4 ** kmer + 1, maximum_position_encoding=len(training_seq[0]), rate=rate)(input)

    model = tf.keras.models.Model(inputs=input, outputs=prediction)
    model.compile(loss='mean_squared_error', optimizer='adam')

    checkpoint_path = "/checkpoints/" + datasetname
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    scores = model.predict(x=testing_seq)

    auc = roc_auc_score(testing_labels, scores)
    print("AUC =", auc)

    ap = average_precision_score(testing_labels, scores)
    print("AP =", ap)