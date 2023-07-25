# MinningTweets_LDA_LSTM_Dic
The code is for minning tweets using LDA, LSTM and vader dictionary
## Description
The folder ['lexicon'](lexicon) contains the dictionary vader that used for sentiment analysis.

The folder ['pics'](pics) contains the result of LDA word cloud and sentiment analysis.

[swelling.py](swelling.py) uses the idea of stream listening based on Scweepy package, with web crawler we can extract tweets that related to our keywords.

[text_cleaning.py](text_cleaning.py) is used for text cleaning. the text cleaning work includes removing 5 different kinds of typical texts that show in tweet: the url, which is usually appear in the tweets in the form of http:// and „.com“; the function of „@ someone“; the stop words, like single number or meaningless letter; some automatic words like „reply“ and „retweets“, and the keywords it self as well its synomyms.

[Language_Model.py](Language_Model.py) is used for minning the topics with LDA.

[RNN_IURT.ipynb](RNN_IURT.ipynb) is used for training and classify the tweets that relating to different topics based on LSTM, such as "self driving" and "law", the training dataset can be reached in the following limk: [self_driving](https://drive.google.com/file/d/1sQ7Wi643bvsda_m_k6nuKgPYhkCB1igK/view?usp=drive_link) and [law](https://drive.google.com/file/d/1dB8VPUykNdjepeXIT6wLel7tpEfw3Rur/view?usp=drive_link)

[vaderSentiment.py](vaderSentiment.py) is used for sentiment analysis using the vader lexicon.

## Results
LDA in Word cloud:
![](/pics/image.png)
LDA in statistics:
[link](/pics/LDA_Vis.html)

Sentiment Analysis:
![](/pics/Senti_general.png)

