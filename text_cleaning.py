import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import traceback
import pymongo
import collections
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora, models, similarities
import gensim

'''删除推文中的url'''
def clean_url(text):
    sentences = text.split(' ')
    # 处理http://类链接
    url_pattern = re.compile(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b', re.S)
    # 处理无http://类链接
    domain_pattern = re.compile(r'(\b)*(.*?)\.(com|cn)')
    if len(sentences) > 0:
        result = []
        for item in sentences:
            text = re.sub(url_pattern, '', item)
            text = re.sub(domain_pattern,'', text)
            result.append(text)
        return ' '.join(result)
    else:
        return re.sub(url_pattern, '', sentences)

'''删除推文中的at'''
def clean_at(text):
    at_pattern = re.compile('@\S*', re.S)
    text = re.sub(at_pattern, '', text)
    return text.strip()

'''删除推文中的中文（which is like 转推、回复等字眼）'''
def clean_NonEn(text):
    text = re.sub('[\u4e00-\u9fa5]','',text)
    return text.strip()

'''删除推文中孤立的数字 无意义的字母或单词-stop_words'''
def clean_singleC(text):
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    text = text.lower()
    text = tokenizer.tokenize(text)  # Split into words.

    # Remove numbers, but not words that contain numbers.
    text = [token for token in text if not token.isdigit()]
 #   text = [token for token in text if token not in stop_words]

    # Lemmatize all words in documents.

    return text
'''并将句子转换为单词'''

def cleaning_text(texts):
    cleaned_text = []
    id = []
    i = 1
    for text in texts:
        if not pd.isna(text):
            text = clean_url(text)
            text = clean_at(text)
            text = clean_NonEn(text)
            text = clean_singleC(text)
            print(text)
            cleaned_text.append(text)
            id.append(str(i))
            i = i + 1 
    return cleaned_text,id

'''词云类'''
class MongoConn():
    def __init__(self, db_name):
        try:
            url = '127.0.0.1:27017'
            self.client = pymongo.MongoClient(url, connect=True)
            self.db = self.client[db_name]
        except Exception as e:
            print ('连接mongo数据失败!')
            traceback.print_exc()

    def destroy(self):
        self.client.close()

    def getDb(self):
        return self.db

    def __del__(self):
        self.client.close()

'''词云生成器'''
class cloudProducer():

    def __init__(self):

        self.mon = MongoConn('ANXIETY')
        self.db = self.mon.getDb()

    def produce_Cloud(self, texts, num, keywords):
        #main page
        words_dump = []

        texts, id = cleaning_text(texts)

        for text in texts:
            words_dump = words_dump + text


        cloud = collections.Counter(words_dump).most_common(num)
        cloud = sorted(cloud,key=lambda t: t[1],reverse=True)
        
        re_cloud = []
        for word in cloud:
            if word[0] not in keywords:
                re_cloud.append(word)

        word_dic = {}
        for word in re_cloud:
            word_dic[word[0]] = word[1]
        # 建立词袋，并去除本次project的关键字 ai ethics

        wordcloud = WordCloud(background_color = 'white').fit_words(word_dic)  
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()        
        
        return re_cloud[0:num], word_dic

def LDA_text(texts, keywords):
    texts, id = cleaning_text(texts)
    texts = [[word for word in text if word not in keywords] for text in texts]
    dictionary = corpora.Dictionary(texts) # 根据现有数据生成词典
    dictionary.filter_n_most_frequent(2)
    # 去除频率最高的两个单词：ai, ethic
    corpus = [dictionary.doc2bow(sentence) for sentence in texts] # 对每句话，统计每个词语的频数，组成词袋模型
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)
    return lda



pd_reader = pd.read_csv("AI ethics_2021-1-1_2021-12-31.csv")
# pd_reader_2 = pd.read_csv("./outputs/chatGPT_GPT-3_2022-09-01_2022-12-31.csv")
# pd_reader_3 = pd.read_csv("./outputs/chatGPT_GPT-3_2023-1-16_2023-1-31.csv")
# pd_reader_4 = pd.read_csv("./outputs/chatGPT_GPT-3_2023-2-1_2023-2-16.csv")

# pd_reader = pd.concat([pd_reader_1,pd_reader_2],axis=0)

text, id =  cleaning_text(pd_reader.Embedded_text)
cleaned_text = pd.DataFrame(index=id, data=text)


cleaned_text.to_csv("Cleaned_outputs_Withstopwords_AIethics_2021.csv",index=False)

time = pd.DataFrame(pd_reader[pd_reader.Embedded_text.notna()].Timestamp)
time.to_csv("Cleaned_outputs_Withstopwords_AIethics_2021_time.csv",index=False)
except_keywords = set('ai ethics'.split())

# # cp = cloudProducer()
# # cloud, word_dic = cp.produce_Cloud(pd_reader.Embedded_text,15,except_keywords)
# # 词云模型
# # print(cloud)
# # 统计高频词汇 %无意义actually

# lda = LDA_text(pd_reader.Embedded_text, except_keywords)

# for topic in lda.print_topics(num_topics=20, num_words=4):
#     print(topic[1])




