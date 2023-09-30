## sentiment analysis
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import pandas as pd

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

def sentimentanalyzer(sent):
    encoded_input = tokenizer(sent, return_tensors='pt')
    output = model(**encoded_input)
    score = output[0][0].detach().numpy()
    score = softmax(score)

    return score


texts = pd.read_excel('c:/python/datascience/sentiment/texts_adollshouse_line.xlsx')
sents = list(texts["sentences"])

scores = np.zeros(shape=(3,))
for sent in sents:
    score = sentimentanalyzer(sent)
    scores = np.vstack([scores, score])

texts.loc[:, ['negative', 'neutral', 'positive']] = scores[1:]

texts.to_excel(f'c:/python/datascience/sentiment/Sentiment_by_line.xlsx', index=0)



### graphs
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns; sns.set()

def graph1(df, measure, points, title):
    positive = list(df['positive'])
    negative = list(df['negative'])

    # moving average
    sentiments_df = pd.DataFrame(positive, columns=["positive"])
    sentiments_df["negative"] = pd.DataFrame(negative)
    sentiments_df["diff"] = sentiments_df["positive"]-sentiments_df["negative"]

    # sentiments_df["pos_ma"] = sentiments_df["positive"].rolling(window=int(len(sentiments_df) / 10), center=True).mean()
    # sentiments_df["neg_ma"] = sentiments_df["negative"].rolling(window=int(len(sentiments_df) / 10), center=True).mean()
    sentiments_df["diff_ma"] = sentiments_df["diff"].rolling(window=int(len(sentiments_df) / 10), center=True).mean()
    sentiments_df["diff_cm"] = sentiments_df["diff"].cumsum()

    del sentiments_df['positive'], sentiments_df['negative'], sentiments_df["diff"]

    fig1 = plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=sentiments_df[measure])
    ax.set(xlabel='Line Index', ylabel='Valence', title=f'{title}')
    acts = np.where(df.character == 'ACT')
    for i in acts[0]:
        plt.axvline(i, linewidth=0.5, linestyle='dashed', color='k')
        plt.text(i, round(np.min(sentiments_df[f'{measure}']), 2), df.sentences[i], ha='right')
    plt.axhline(texts["diff"].mean(), linestyle='dashed', color='k')
    # plt.axhline(texts.negative.mean(), linestyle='dashed', color='orange')

    for r in range(len(points)):
        if points[r].type == 'start':
            color = points[r].color
            text = points[r].annotation
            start = [i for i in range(len(df)) if points[r].line in df.sentences[i]][0]
            end = [i for i in range(len(df)) if points[(r+1)].line in df.sentences[i]][0]
            plt.axvspan(start, end, facecolor=color, alpha=0.2)
            plt.text(start, round(np.max(sentiments_df[f'{measure}']), 2), text, size=15)

def graph2(df, name1, name2, measure, points, title):
    def compute(df, name):
        sentiments_df = df[df.character==name].reset_index(drop=True)
        sentiments_df['diff']= sentiments_df["positive"]-sentiments_df["negative"]

        # moving average
        sentiments_df["pos_ma"] = sentiments_df["positive"].rolling(window=int(len(sentiments_df) / 10), center=True).mean()
        sentiments_df["neg_ma"] = sentiments_df["negative"].rolling(window=int(len(sentiments_df) / 10), center=True).mean()
        sentiments_df["diff_ma"] = sentiments_df["diff"].rolling(window=int(len(sentiments_df) / 10), center=True).mean()
        sentiments_df["diff_cm"] = sentiments_df["diff"].cumsum()
        return sentiments_df

    n1 = compute(df, name1)
    n2 = compute(df, name2)
    if measure == 'diff_cm':
        n2[measure] = n2[measure]*len(n1)/len(n2)

    fig1 = plt.figure(figsize=(8, 5))
    ax = sns.lineplot(x=n1["index"], y=n1[measure], data=n1, color='darkcyan')
    ax = sns.lineplot(x=n2["index"], y=n2[measure], data=n2, color='darksalmon')

    nl1 = mlines.Line2D([], [], color='darkcyan', ls='-', label=name1)
    nl2 = mlines.Line2D([], [], color='darksalmon', ls='-', label=name2)
    # etc etc
    plt.legend(handles=[nl1, nl2])
    ax.set(xlabel='Line Index', ylabel='Valence', title=title)

    acts = np.where(df.character == 'ACT')
    for i in acts[0]:
        plt.axvline(i, linewidth=0.5, linestyle='dashed', color='k')
        plt.text(i, round(np.min(n1[measure]), 2), df.sentences[i], ha='left')

    for r in range(len(points)):
        if points[r].type == 'start':
            color = points[r].color
            text = points[r].annotation
            start = [i for i in range(len(df)) if points[r].line in df.sentences[i]][0]
            end = [i for i in range(len(df)) if points[(r+1)].line in df.sentences[i]][0]
            plt.axvspan(start, end, facecolor=color, alpha=0.3)
            plt.text(start, round(np.max(n1[measure]), 2), text, size=15)

def graph3(df, name1, name2, points, title):
    def compute(df, name):
        sentiments_df = df[df.character==name].reset_index(drop=True)
        sentiments_df["diff_cm"] = sentiments_df["diff"].cumsum()
        return sentiments_df

    n1 = compute(df, name1)
    n2 = compute(df, name2)
    n2['diff_cm_weighted'] = n2['diff_cm']*len(n1)/len(n2)

    fig1 = plt.figure(figsize=(10, 6))
    ax = sns.lineplot(x=n1["index"], y=n1['diff_cm'], data=n1, color='darkcyan')
    ax = sns.lineplot(x=n2["index"], y=n2['diff_cm_weighted'], data=n2, color='coral')
    ax = sns.lineplot(x=n2["index"], y=n2['diff_cm'], data=n2, color='coral', linestyle='dashed')


    nl1 = mlines.Line2D([], [], color='darkcyan', ls='-', label=name1)
    nl2 = mlines.Line2D([], [], color='coral', ls='--', label=name2)
    nl3 = mlines.Line2D([], [], color='coral', ls='-', label=f'{name2}-weighted')

    # etc etc
    plt.legend(handles=[nl1, nl2, nl3])
    ax.set(xlabel='Line Index', ylabel='Valence', title=title)

    acts = np.where(df.character == 'ACT')
    for i in acts[0]:
        plt.axvline(i, linewidth=0.5, linestyle='dashed', color='k')
        plt.text(i, round(np.min(n1['diff_cm']), 2), df.sentences[i], ha='left')

    for r in range(len(points)):
        if points[r].type == 'start':
            color = points[r].color
            text = points[r].annotation
            start = [i for i in range(len(df)) if points[r].line in df.sentences[i]][0]
            end = [i for i in range(len(df)) if points[(r+1)].line in df.sentences[i]][0]
            plt.axvspan(start, end, facecolor=color, alpha=0.3)
            plt.text(start, round(np.max(n1['diff_cm']), 2), text, size=15)

def graph4(df, name1, name2, measure, points, title):
    def compute(sentiments_df, length):
        # moving average
        sentiments_df["pos_ma"] = sentiments_df["positive"].rolling(window=int(len(sentiments_df) / 10), center=True).mean()
        sentiments_df["neg_ma"] = sentiments_df["negative"].rolling(window=int(len(sentiments_df) / 10), center=True).mean()
        sentiments_df["diff_ma"] = sentiments_df["diff"].rolling(window=int(len(sentiments_df) / 10), center=True).mean()
        sentiments_df["diff_cm"] = sentiments_df["diff"].cumsum()
        return sentiments_df

    n1 = df[df.character==name1].reset_index(drop=True)
    n2 = df[df.character==name2].reset_index(drop=True)
    if len(n1)>len(n2):
        length = len(n2)
    else:
        length = len(n1)
    n1 = compute(n1, length)
    n2 = compute(n2, length)
    if measure in ['pos_cm', 'ntrl_cm', 'neg_cm', 'diff_cm']:
        n2[measure] = n2[measure]*len(n1)/len(n2)

    point = [i for i in range(len(df)) if points[0].line in df.sentences[i]][0]
    fig1 = plt.figure(figsize=(8, 5))
    ax = sns.lineplot(x="index", y=f"{measure}", data=n1[n1["index"]>point], color='darkcyan')
    ax = sns.lineplot(x="index", y=f"{measure}", data=n2[n2["index"]>point], color='darksalmon')
    nl1 = mlines.Line2D([], [], color='darkcyan', ls='-', label=name1)
    nl2 = mlines.Line2D([], [], color='darksalmon', ls='-', label=name2)
    # etc etc
    plt.legend(handles=[nl1, nl2])
    ax.set(xlabel='Sentence Index', ylabel='Valence', title=f'{title}')

    for r in range(len(points)):
        if points[r].type == 'climax':
            text = points[r].annotation
            posx = [i for i in range(len(df)) if points[r].line in df.sentences[i]][0]
            plt.axvline(posx, linewidth=0.5, linestyle='dashed', color='k')
            plt.text(posx, round(np.min(n1[f'{measure}']), 2), text, size=15)

path = 'c:/python/datascience/sentiment/plays/'
txt_list = [y for y in os.listdir(path) if 'txt' in y]
filename = txt_list[0]
event = pd.read_excel('c:/python/datascience/sentiment/pevent.xlsx')
pevent = event[event["play"] == filename]
points = [row for _, row in pevent.iterrows()]

texts = pd.read_excel('c:/python/nlp/sentiment/Sentiment_by_line.xlsx')
texts = pd.read_excel('c:/python/nlp/sentiment/Sentiment_by_sentence.xlsx')

plt.plot(texts["index"],texts["diff"])
plt.show()


texts["diff"]= texts["positive"]-texts["negative"]

graph1(texts, "diff_ma", points, 'A Doll\'s House (Moving Average)')
graph1(texts, "diff_cm", points, 'A Doll\'s House (Cumulative)')

graph2(texts, 'NORA', 'HELMER', "diff_ma", points, 'A Doll\'s House-NORA vs. HELMER (Difference)')
graph2(texts, 'NORA', 'HELMER', "pos_ma", points, 'A Doll\'s House-NORA vs. HELMER (Positive)')
graph2(texts, 'NORA', 'HELMER', "neg_ma", points, 'A Doll\'s House-NORA vs. HELMER (Negative)')
graph2(texts, 'MRS LINDE', 'KROGSTAD', "diff_cm", points, 'A Doll\'s House-NORA vs. HELMER (Cumulative)')
graph2(texts, 'MRS LINDE', 'KROGSTAD', "diff_ma", points, 'A Doll\'s House-NORA vs. HELMER (Difference)')

graph3(texts, 'NORA', 'HELMER', points, 'A Doll\'s House-NORA vs. HELMER (Cumulative)')

graph4(texts, 'NORA', 'HELMER', "diff_cm", points, 'A Doll\'s House (Climax)-NORA vs. HELMER (Cumulative)')
graph4(texts, 'NORA', 'HELMER', "diff_ma", points, 'A Doll\'s House (Climax)-NORA vs. HELMER (Moving Average)')

texts["diff"]= texts["positive"]-texts["negative"]
words = [len(x.split(" ")) for x in texts.sentences]
texts["wordlen"] = [len(x.split(" ")) for x in texts.sentences]
texts.groupby("character").mean()
texts.groupby("character").count()
texts.mean()
texts.std()


