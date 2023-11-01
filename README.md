# phase5
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import nltk
import os
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
email_df=pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
email_df.head()
email_df.info()
column_to_delete=[name for name in email_df.columns if name.startswith('Unnamed')]
email_df.drop(columns=column_to_delete,inplace=True)
email_df.rename(columns=dict({"v1":"target","v2":"message"}),inplace=True)
email_df.tail()
email_df.isnull().sum()
print("Total duplicated records in dataset are : {}".format(email_df.duplicated().sum()))
def target_mapper(text):
    return 0 if text=='spam' else 1

email_df["target"]=email_df['target'].apply(func=target_mapper)
import nltk
nltk.download('punkt') #download punctuation
nltk.download('stopwords') #download stopwords
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import tokenize
STOPWORDS=stopwords.words("english")
def message_tranformation(text):
    text=text.strip()
    text=text.lower()
words=tokenize.word_tokenize(text)
 stemmer=PorterStemmer()
filtered_words=[stemmer.stem(word) for word in words if word not in STOPWORDS and word.isalnum()]
    transformed_text=" ".join(filtered_words)
    return transformed_text
email_df["transformed_message"]=email_df["message"].apply(message_tranformation)
email_df.head()
email_df.drop(columns="message",inplace=True)
from wordcloud import WordCloud
wc=WordCloud(width=1000,height=1000,min_font_size=8,background_color='white')
spam_wc=wc.generate(email_df[email_df["target"]==0]["transformed_message"].str.cat(sep=" "))
plt.figure(figsize=(20,10))
plt.imshow(spam_wc)
plt.show()
ham_wc=wc.generate(email_df[email_df["target"]==1]["transformed_message"].str.cat(sep=" "))
plt.figure(figsize=(20,10))
plt.imshow(ham_wc)
plt.show()
spam_corpus=list()
for msg in email_df[email_df['target']==0]["transformed_message"].to_list():
    for word in msg.split():
        spam_corpus.append(word)
len(spam_corpus)
from collections import Counter
spam_top_50_common_words=pd.DataFrame(Counter(spam_corpus).most_common(50))
print(spam_top_50_common_words)
ham_corpus=list()
for msg in email_df[email_df['target']==1]["transformed_message"].to_list():
    for word in msg.split():
        ham_corpus.append(word)
len(ham_corpus)
ham_top_50_common_words=pd.DataFrame(Counter(ham_corpus).most_common(50))
print(ham_top_50_common_words)
from sklearn.feature_extraction.text import CountVectorizer
cVector=CountVectorizer()
x=cVector.fit_transform(email_df["transformed_message"]).toarray()
y=email_df['target']
plt.pie(y.value_counts().values,labels=["Not Spam","Spam"],autopct="%0.2f%%")
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
def evaluate_model_performance(model,x_test,y_test):
    y_pred=model.predict(x_test)
    print("Accurary Score : {}".format(np.round(accuracy_score(y_test,y_pred)*100,decimals=2)))
    print("Precision Score : {}".format(np.round(precision_score(y_test,y_pred)*100,decimals=2)))
    print("Recall Score : {}".format(np.round(recall_score(y_test,y_pred)*100,decimals=2)))
    print("F1 Score : {}".format(np.round(f1_score(y_test,y_pred)*100,decimals=2)))
    cm=confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,fmt="d",annot=True,cmap="rainbow")
    plt.show()
    print("Classification Report****************")
    print(classification_report(y_test,y_pred))
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
models = {
    "lr":LogisticRegression(),
    "nb":MultinomialNB(),
    "svm":SVC(),
    "knn":KNeighborsClassifier(),
    "cart":DecisionTreeClassifier(),
    "rf":RandomForestClassifier(),
    "ad":AdaBoostClassifier(),
    "gb":GradientBoostingClassifier(),
    "xgbc":XGBClassifier()
}
oversampler = RandomOverSampler()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_scores=list()
for model_name, model in models.items():
X_resampled, y_resampled = oversampler.fit_resample(x, y)
scores = cross_val_score(model, X_resampled[:500], y_resampled[:500], cv=cv, scoring="f1_micro")
    print(model_name," : ",np.round(np.mean(scores)*100,decimals=2))
    model_scores.append(scores)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(model_scores)
ax.set_xticklabels(models.keys())
plt.show()
model=MultinomialNB()
model.fit(x_train,y_train)
print("Model Training score : ",model.score(x_train,y_train))
evaluate_model_performance(model,x_test,y_test)
