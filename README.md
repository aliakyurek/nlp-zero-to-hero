## NLP
Step by step NLP examples using PyTorch and Lightning
1. LSTM Character Generation with LSTMCell [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakyurek/nlp/blob/main/LSTM_Character_Generation_with_LSTMCell.ipynb) 
  * Character RNN that's trained with a covid_19 faq file
  * After training, it generates text based on a prime input.
  * Generated text and loss values are recorded to and observed
2. LSTM Character Generation with LSTM [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakyurek/nlp/blob/main/LSTM_Character_Generation_with_LSTM.ipynb) 
  * Same as above but this time it uses LSTM block
3. Language translation with LSTM [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakyurek/nlp/blob/main/Language_translation_with_LSTM.ipynb) 
  * Seq2Seq training and inference using LSTM layer
4. Language translation with GRU [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakyurek/nlp/blob/main/Language_translation_with_GRU.ipynb) 
  * Same as above but utilizes GRU instead of LSTM and more input used for GRU and decoder outputs
5. Language translation with Bahdanau attention [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakyurek/nlp/blob/main/Language_translation_with_GRU_and_Bahdanau.ipynb)
  * Same as above but utilizes Bahdanau attention (not self attention) using bidirectional GRU
6. Language translation with padding optimization
7. Language translation with transformers
  * Basic implementation of transformer architecture for translation problem
    
References
* seq2seq examples by Ben Trevett https://github.com/bentrevett/pytorch-seq2seq
* Deep Learning Course by Prof.Sebastian Raschka https://www.youtube.com/watch?v=1nqCZqDYPp0&list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51
* Deep Learning Tutorials by UvA https://uvadlc-notebooks.readthedocs.io/en/latest/index.html 
* TSAI-DeepNLP-END2.0 by extensive-nlp https://github.com/extensive-nlp/TSAI-DeepNLP-END2.0/

    
