import tensorflow as tf
import numpy as np

with open("data.txt", "r", encoding='utf-8') as f:
    text = f.read()   
chars = sorted(list(set(text)))
vocab_size = len(chars)
embedding_dim = 256 
rnn_units = 1024
char2idx = {u: i for i, u in enumerate(chars)}
idx2char = np.array(chars)
text_as_int = np.array([char2idx[c] for c in text])
seq_length = 40
examples_per_epoch = len(text)//(seq_length + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()

inputs = tf.keras.Input(batch_shape=(BATCH_SIZE, None), dtype=tf.int32)

x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
x = tf.keras.layers.GRU(
    rnn_units,
    return_sequences=True,
    stateful=False,
    recurrent_initializer='glorot_uniform'
)(x)
outputs = tf.keras.layers.Dense(vocab_size)(x)

model = tf.keras.Model(inputs, outputs)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

EPOCHS = 20
history = model.fit(dataset, epochs=EPOCHS,steps_per_epoch=100)



def generate_text(model, start_string, num_generate=300):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0) 

    text_generated = []

    temperature = 0.7

    for i in range(num_generate):
        predictions = model(input_eval)  
        predictions = predictions[:, -1, :]  
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()

      
        text_generated.append(idx2char[predicted_id])

        
        input_eval = tf.expand_dims([predicted_id], 0)

    return start_string + ''.join(text_generated)

user_input = input("Starting phrase for the poem: ")
generated_poem = generate_text(model, start_string=user_input, num_generate=300)

print(generated_poem)
