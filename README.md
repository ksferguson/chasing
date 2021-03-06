# Chasing the Flagellum
### Rare Words for Text Generation

A language models is a Recurrent Neural Network typically trained on a large corpus of text to learn the probabilities of words occurring together to form various sentences, questions, and other expressions.

Aside from say the top few hundred words, most words are very infrequent, making it challenging for a language model to predict the next word. This project explores how we can better generate the next word in an expression by applying technique(s) designed to improve the performance of rare words.  

## Requisites

#### Base environment

```bash
#assumes Anaconda installed with 'base' environment
. activate base
conda create --name chasing python=3.6
. activate chasing
conda install pytorch=0.1.12 -c soumith
conda install cython
pip install streamlit
```

#### Project Code

```bash
# Clone project
git clone https://github.com/ksferguson/chasing
```

Note: This project started from a copy of https://github.com/salesforce/awd-lstm-lm, pulled with Tag: PyTorch==0.1.12


## Train Model
### Initial training
```bash
python main.py --batch_size 40 --data data/penn --dropouti 0.4 --seed 28 --epoch 300 --save PTB.pt
```
You may want to copy the model PTB.pt to save the state before proceeding to second stage training.

### Second Stage training
```bash
python finetune.py --batch_size 40 --data data/penn --dropouti 0.4 --seed 28 --epoch 300 --save PTB.pt
```

### Generate
See the args in generate.py for full list of options:
```bash
python generate.py --data data/penn --checkpoint PTB.pt --words 10 --temperature 1.0 --log-interval 10 --cuda --warmup --context 'i want to get lunch this week <eos> are you free'
```
