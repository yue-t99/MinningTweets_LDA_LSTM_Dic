import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import traceback
import pymongo
import collections
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from gensim import corpora, models, similarities
import gensim
import numpy as np
import nltk 
from nltk import word_tokenize #分词函数
from nltk.corpus import sentiwordnet as swn #得到单词情感得分
import string #本文用它导入标点符号，如!"#$%& 
import pyLDAvis
from pyLDAvis import gensim as gensim_vis


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
        words_dump = texts
        # words_dump = []

        # texts, id = cleaning_text(texts)

        # texts = [[word for word in text if word not in keywords and not str(word)=='nan'] for text in texts]
        # for text in texts:
        #     words_dump = words_dump + text


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

def LDA_text(texts, keywords, num_topics):
#    texts, id = cleaning_text(texts)
    texts = [[word for word in text if word not in keywords and not str(word)=='nan'] for text in texts]
    dictionary = corpora.Dictionary(texts) # 根据现有数据生成词典
    # dictionary.filter_n_most_frequent(2)
    # 去除频率最高的两个单词：ai, ethic
    corpus = [dictionary.doc2bow(sentence) for sentence in texts] # 对每句话，统计每个词语的频数，组成词袋模型
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    vis_data = gensim_vis.prepare(lda, corpus, dictionary, sort_topics=False) 
    pyLDAvis.save_html(vis_data, "LDA_Vis.html")
    return lda

def text_score(text):
    #create单词表
    #nltk.pos_tag是打标签
    text = [i for i in text if not str(i)=='nan']
    ttt = nltk.pos_tag(text)
    word_tag_fq = nltk.FreqDist(ttt)
    wordlist = word_tag_fq.most_common()

    #变为dataframe形式
    key = []
    part = []
    frequency = []
    for i in range(len(wordlist)):
        key.append(wordlist[i][0][0])
        part.append(wordlist[i][0][1])
        frequency.append(wordlist[i][1])
    textdf = pd.DataFrame({'key':key,
                      'part':part,
                      'frequency':frequency},
                      columns=['key','part','frequency'])

    #编码
    n = ['NN','NNP','NNPS','NNS','UH']
    v = ['VB','VBD','VBG','VBN','VBP','VBZ']
    a = ['JJ','JJR','JJS']
    r = ['RB','RBR','RBS','RP','WRB']

    for i in range(len(textdf['key'])):
        z = textdf.iloc[i,1]

        if z in n:
            textdf.iloc[i,1]='n'
        elif z in v:
            textdf.iloc[i,1]='v'
        elif z in a:
            textdf.iloc[i,1]='a'
        elif z in r:
            textdf.iloc[i,1]='r'
        else:
            textdf.iloc[i,1]=''
            
        #计算单个评论的单词分数
    score_pos = []
    score_neg = []
    for i in range(len(textdf['key'])):
        m = list(swn.senti_synsets(textdf.iloc[i,0],textdf.iloc[i,1]))
        s_pos = 0
        s_neg = 0
        ra = 0
        if len(m) > 0:
            for j in range(len(m)):
                s_pos += m[j].pos_score()/(j+1)
                s_neg += m[j].neg_score()/(j+1)
               # ra += 1/(j+1)
            score_pos.append(s_pos/len(m))
            score_neg.append(s_neg/len(m))
        else:
            score_pos.append(0)
            score_neg.append(0)
    return sum(score_pos), sum(score_neg)
    # return pd.concat([textdf,pd.DataFrame({'score':score})],axis=1)

def sentiWN(texts):
    score_pos = []
    score_neg = []
    for text in texts:
        pos, neg = text_score(text)
        score_pos.append(pos)
        score_neg.append(neg)
    return score_pos, score_neg

def getTimestamp(time):
    t_list = []
    for t in time:
        t_list.append(t.timestamp())  
    return pd.DataFrame(index=range(len(t_list)), data=t_list)


data = pd.read_csv("Cleaned_outputs_AIethics_2021.csv",index_col = False)
# data_22 = pd.read_csv("./outputs/outputs_AIethics_2022.csv",index_col = False)
# data = pd.concat([data_21, data_22],axis=0)

texts = data.values.tolist()
# except_keywords = set('ChatGPT GPT-3 GPT chatgpt gpt ai gpt3'.split())
except_keywords = set('ai ethics artificial intelligence 100daysofcode ethical aiethics artificialintelligence timnit gebru via'.split())
# except_keywords = set(''.split())


'''SentiWordNet for sentiment analysis'''
# time = pd.read_csv("./outputs/_AIethics_outputs_2020_time.csv",index_col=False)
# time_datetime = pd.to_datetime(time.Timestamp, format="%Y-%m-%d")
# datatime_num = getTimestamp(time_datetime)
# # SWN_test = pd.read_csv("./outputs/outputs_0101_1231_SWN.csv",index_col=False)
# SWN_pos, SWN_neg = sentiWN(texts)

# id = range(len(SWN_pos))
# df_Swn_pos = pd.DataFrame({'pos':SWN_pos})
# df_Swn_neg = pd.DataFrame({'neg':SWN_neg})
# df_Swn_pos = pd.concat([df_Swn_pos, time_datetime,datatime_num],axis=1)
# df_Swn_neg = pd.concat([df_Swn_neg, time_datetime,datatime_num],axis=1)

# df_Swn_pos = df_Swn_pos.sort_values(by=[0], ascending=False)
# df_Swn_neg = df_Swn_neg.sort_values(by=[0], ascending=False)

# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
# ax1.plot(df_Swn_pos['Timestamp'], df_Swn_pos['pos'], lw = 1, color = 'palevioletred', label = "pos")
# ax1.plot(df_Swn_neg['Timestamp'], df_Swn_neg['neg'], lw = 1, color = 'lightblue', label = "neg")
# # ax1.plot(SWN_test['Timestamp'], SWN_test['0'], lw = 1, color = 'lightblue', label = "neg")
# ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
# plt.xticks(pd.date_range(time_datetime[0],time_datetime[len(time_datetime)-1],freq='2m'))
# ax1.set_ylabel('emotion')
# ax1.set_xlabel('time')
# plt.title("pos-neg in 2022")
# plt.gcf().autofmt_xdate()
# plt.legend(['positive','negative'])
# plt.show()
# print(SWN_pos)

# id = range(len(SWN_test))
# df_SWN = pd.DataFrame(index = id, data = SWN_test)
# df_SWN.to_csv("./outputs/outputs_0101_1231_SWN.csv", index=False)

'''LDA for topics extraction'''
num_topics = 8
num_words = 30
lda = LDA_text(texts, except_keywords, num_topics)
re = lda.show_topics(num_topics=num_topics, num_words=num_words, log=False, formatted=True)
for topic in lda.print_topics(num_topics=num_topics, num_words=10):
    print(topic[1])
for topic in re:
    topic_temp = topic[1].replace('+',' ')
    topic_temp = topic_temp.replace('*',' ')
    topic_temp = topic_temp.replace('"','')
    w_f = topic_temp.split()
    words = ''
    for i in range(num_words):
        freq = int(1000*float(w_f[i*2]))
        words = words + freq*(w_f[i*2+1]+' ')
    
    worldC = cloudProducer()
    re_wc, wc_dic = worldC.produce_Cloud(texts=words.split(), num=num_words, keywords=except_keywords)


# '''Word Cloud for word frequency analysis'''
# worldC = cloudProducer()
# re_wc, wc_dic = worldC.produce_Cloud(texts=texts, num=40, keywords=except_keywords)

print('success')