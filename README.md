CodeClauseInternship_PlagirismChecker:

The Plagiarism Checker project is a natural language processing (NLP) project that uses machine learning to detect plagiarism in text. The project takes two pieces of text as input: a suspected plagiarized text and a reference text. The project then uses a variety of NLP techniques to compare the two texts and determine the level of plagiarism.
I used the pretrained microsoft/mpnet-base model and fine-tuned in on a 1B sentence pairs dataset. 
The model is intented to be used as a sentence and short paragraph encoder. Given an input text, it ouptuts a vector which captures the semantic information. The sentence vector is used for sentence similarity tasks.
TRAINING PROCEDURE:

Pre-training:
I have used the pretrained microsoft/mpnet-base model.

Fine-tuning:
I have fine-tuned the model using a contrastive objective. Formally, we compute the cosine similarity from each possible sentence pairs from the batch. then i have apply the cross entropy loss by comparing with true pairs.

Hyper parameters:
I have trained this model on a TPU v3-8. I trained the model during 100k steps using a batch size of 1024 (128 per TPU core). I have used a learning rate warm up of 500. The sequence length was limited to 128 tokens. I have used the AdamW optimizer with a 2e-5 learning rate. The full training script is accessible in this current repository: train_script.py.

Training data:
I have used the concatenation from multiple datasets to fine-tune the model. The total number of sentence pairs is above 1 billion sentences. I have sampled each dataset given a weighted probability which configuration is detailed in the data_config.json file.

INSTRUCTIONS:
1. Clone this repository
2. Navigate to this folder in terminal and type "pip install -r requirements.txt"
3. then after the setup run the command "python Plagarism checker.py"
