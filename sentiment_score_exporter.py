from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import pandas as pd
from scipy.special import softmax

MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)

def sentimentanalyzer(sent):
    encoded_input = tokenizer(sent, return_tensors='pt')
    output = model(**encoded_input)
    score = output[0][0].detach().numpy()
    score = softmax(score)

    return pd.Series(score, index=['negative', 'neutral', 'positive'])


root = './data/'

texts = pd.read_excel(f'{root}/SentbySent.xlsx', index_col=0)

scores = texts['sentences'].apply(sentimentanalyzer)

texts = pd.concat([texts, scores], axis=1)
texts['compound'] = (texts['positive']-texts['negative'])*100
texts.reset_index(level=0, inplace=True)

texts.to_excel(f'{root}/SentbySent_scored.xlsx', index=False)
