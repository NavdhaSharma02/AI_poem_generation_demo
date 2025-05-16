# AI_poem_generation_demo

Just trying to learn how to train an AI model and hence, this is a beginner level AI project that uses a Recurrent Neural Network (RNN) to generate poetic text based on a small training dataset.

## Tech Stack used:-
- Programming language- Python
- library:- Numpy
- RNN:- TensorFlow / Keras
- Python virtual environment

--- 

##  Model Architecture

- **Embedding Layer**: Learns a dense vector representation of characters  
  `embedding_dim = 256`
- **GRU Layer**: Gated Recurrent Unit for learning sequential dependencies  
  `rnn_units = 1024`
- **Dense Layer**: Predicts the next character from the vocabulary

---

## File structure:-
AI_poem_generation_demo
- data.txt         <- A samll dataset containing training data
- generate.py      <- script 
- venv              <-virtual environment

---

## RNN
So I used Tensorflow framework here to build my project.Tesnor flow helps in training deep neural networks .
Key Components OF Tensorflow:-
tf.keras: High-level API for building models quickly.
tf.data: Tools for handling data pipelines efficiently.
tf.Tensor: The main data structure (n-dimensional array).
TensorBoard: For visualizing training progress and debugging.

## How It Works:-
- The "data.txt" file contains short poetic lines.
- The RNN learns the patterns and structure of the text.
- After training, the model can generate new poetic sequences similar in style.
- Text data → Clean + Encode → Batch sequences → Train RNN → Generate poetry

## Drawbacks
- It is not stateful (can't remember anything it process)
- Too less training data
- Too less epoches
- Sequence length too short to capture poetic patterns 
