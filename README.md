# SkipGram

This project is part of an "Advanced NLP" assignment, demonstrating practical applications of word embeddings in 
natural language processing. Focuses on implementing a Skip-gram model for word embeddings using textual data 
from news articles taken from 'newsapi' API. 

## Experiment Parameters
This project experiments with "window size" and "embedding dimension", using <b>10 epochs</b> for training. <br/>
"window size" values used is <b>[2, 3, 4]</b>, "embedding dimension" used is <b>[50, 100, 200]</b>. <br/>
The hyperparameter tuning method applied is <b>Grid Search</b>, 
and "average loss" is used for evaluation to determine the best parameters. <br/>
<b>The smallest mean loss value will be considered the best parameter selection.</b>

## Result 
```python
Best Model: window_size=4, embedding_dim=50, AVG-Loss=10.25992675525068
Words similar to 'government': ['fans', 'all', '000', 'be', 'time']
```
Based on the experiment, the best-performing model was achieved with a
<b>window size of 4</b> and an <b>embedding dimension of 50</b>, 
resulting in the lowest average loss of <b>10.26</b>.
This indicates that a lower embedding dimension might be sufficient to capture semantic relationships.
Additionally, a higher embedding dimension increases computational complexity and resource consumption.




## Usage
Install packages via requirements file
```python
pip install -r requirements.txt
```
Fetching news data via API
```python
python news.py
```
Skipgram training simulation
```python
python app.py
```
Python notebook file also provided in file "SkipGram.ipynb"