# SkipGram

This project is part of an "Advanced NLP" assignment, demonstrating practical applications of word embeddings in 
natural language processing. Focuses on implementing a Skip-gram model for word embeddings using textual data 
from news articles taken from 'newsapi' API. 

## Experiment Parameters
This project experiments with "window size" and "embedding dimension", using 10 epochs for training. <br/>
"window size" values used is [2, 4 ,5], "embedding dimension" used is [50, 100, 200]. <br/>
The hyperparameter tuning method applied is Grid Search, 
and "average loss" is used for evaluation to determine the best parameters. 
The smallest mean loss value will be considered the best parameter selection.

## Result 


## Usage
Fetching news data via API
```python
python news.py
```
Skipgram training simulation
```python
python app.py
```
Python notebook file also provided in file "SkipGram.ipynb"