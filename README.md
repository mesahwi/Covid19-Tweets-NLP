# Covid19-Tweets-NLP

# 1.Problem at hand
Sentiment Analysis. Predict 'sentiment' using tweets.
In this project, I used Naive Bayes, Gradient Boost, XG Boost, CNN, and BERT to predict the sentiment of the tweets. Note that this project is a 'toy project', and therefore far from perfect. There are numerous ways to improve the results, such as fine tuning the models used in the project. 

# 2.Data
## 2.1.From
I downloaded the covid-19-nlp-text-classification dataset from kaggle. It has the columns 'UserName', 'ScreenName', 'Location', 'TweetAt', 'OriginalTweet', and 'Sentiment'. For this project, I tried using only 'OriginalTweet' to predict 'Sentiment'.
## 2.2.Preprocessing
### 2.2.1.Imbalanced Data
Looking at the barplots, we can see right away that the data is a bit unbalanced.
<div>
<img src = "https://user-images.githubusercontent.com/33714067/94115439-ad9b5d80-fe84-11ea-85ef-db2ca89c4821.png">
</div>
I handled this issue by oversampling. I have tried SMOTE and ADASYN on the word embedded texts for creating synthetic dataset, but both methods require too much RAM. Therefore, I am sticking to the simplest method of oversampling, 'RandomOverSampler'. Afterwards, we can see that the data is balanced.
<div>
<img src = "https://user-images.githubusercontent.com/33714067/94116011-737e8b80-fe85-11ea-900b-b97e04ca80b5.png">
</div>

### 2.2.2.Preprocessing
For preprocessing the text data, I used PorterStemmer. It has the disadvantage of being too simple when it comes to stemming. But it is a very light stemmer and I thought it could serve for the purpose of this project. After stemming, I removed stopwords, '$', 'RT', hyperlinks, and '#'. Since tweets are rather short, and sometimes do not contain much text, there were some tweets that disappeared completely after preprocessing. Such instances were removed from the dataset.

# 3.Methods
Here, I introduce the methods used for sentiment analysis.
## 3.1.(Non-Neural Network) Machine Learning
For the 'general' machine learning methods, I used Naive Bayes, Gradient Boost, and XG Boost. For such analyses, I used tf-idf instead of word embeddings.
### 3.1.1.Multinomial Naive Bayes 
Accuracy on training set (Multinomial Bayes): 73.368% <br>
Accuracy on validation set (Multinomial Bayes): 57.638% <br>
Accuracy on test set (Multinomial Bayes): 44.7667% <br>
### 3.1.2.Gradient Boost
With max_depth as 6, and n_estimators as 200, <br>
Accuracy on training set (GBM): 78.1763% <br>
Accuracy on validation set (GBM): 65.2716% <br>
Accuracy on test set: (GBM) 51.5423% <br>
### 3.1.3.XG Boost
With eta as 0.1, lambda as 0.8, max_depth as 6, and n_estimators as 200, <br>
Accuracy on training set (XGBoost): 79.556% <br>
Accuracy on validation set (XGBoost): 66.0876% <br>
Accuracy on test set (XGBoost): 52.8342% <br>

## 3.2.CNN
For CNN, I used word2vec embeddings. Each word was embedded to a 300 dimensional vector. The model was trained for 32 epoches, with min_count 10. Filter sizes for CNN were 3,4,5, with num_filters 512. Adam optimizer with caterogical cross entropy loss was used. The summary of the model is as follows :
<div>
<img src = "https://user-images.githubusercontent.com/33714067/94118853-0cfb6c80-fe89-11ea-9c4b-687e91ed50bd.png">
</div>

Training this model for 8 epochs, I ended up with <br>
loss: 0.4927, accuracy: 0.8173, val_loss: 0.2857, val_accuracy: 0.9430
<div>
<img src = "https://user-images.githubusercontent.com/33714067/94121668-93657d80-fe8c-11ea-932d-13e52ed21deb.png" width=450 height=350>
<img src = "https://user-images.githubusercontent.com/33714067/94121693-99f3f500-fe8c-11ea-9536-fa6fe3f57a69.png" width=450 height=350>
</div>

On the test set, I ended up with loss: 1.2137805223464966, and accuracy: 0.5367782711982727.

## 3.3.BERT
I used the 'bert-base-uncased' model from HuggingFace. In this project, BERT was used on the preprocessed texts. I believe using BERT on unpreprocessed texts could result in better results. The summary of the model is as follows : 
<div>
<img src = "https://user-images.githubusercontent.com/33714067/94119695-38328b80-fe8a-11ea-9875-81c2107727a7.png">
</div>

Training BERT for 3 epochs, I was given <br>
loss: 0.9181, accuracy: 0.6691, val_loss: 0.6060, val_accuracy: 0.7841
<div>
<img src = "https://user-images.githubusercontent.com/33714067/94119719-41235d00-fe8a-11ea-8f39-a1cbaac98bc9.png" width=450 height=350>
<img src = "https://user-images.githubusercontent.com/33714067/94120083-ad05c580-fe8a-11ea-897d-3fc153ff931e.png" width=450 height=350>
</div>

On the test set, I ended up with loss: 0.940627932548523, and accuracy: 0.6538360118865967. <br>

Edit : Using BERT with raw texts did boost the performance to loss: 0.6948069930076599, and accuracy: 0.7484840750694275.

# 4.Further Directions
As mentioned above, there are 2 main ways to improve the model. The first is to preprocess more carefully, perhaps using other stemmers, and using other embedding methods that take into account subwords. The second way is to fine tune the models used above. 
