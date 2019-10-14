from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from io import StringIO
from warnings import simplefilter
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

text1 =  "This company refuses to provide me verification and validation of debt per my right under the FDCPA. I do not believe this debt is mine."
text2 =  "I am disputing the inaccurate information the Chex-Systems has on my credit report. I initially submitted a police report on XXXX/XXXX/16 and Chex Systems only deleted the items that I mentioned in the letter and not all the items that were actually listed on the police report. In other words they wanted me to say word for word to them what items were fraudulent. The total disregard of the police report and what accounts that it states that are fraudulent. If they just had paid a little closer attention to the police report I would not been in this position now and they would n't have to research once again. I would like the reported information to be removed : XXXX XXXX XXXX"
text3 =  "I am disputing with my student loan car paypal"
text4 =  "I am disputing with my mortgage confusing misleading term"
text5 =  "adding money"

import warnings
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Import data


#df = pd.read_csv('pru4.csv',sep=",",index_col=None,low_memory=False)
#df = pd.read_csv('Consumer_Complaints10k.csv',low_memory=False)a


df=joblib.load("entrenamiento_LogisticRegression.joblib")
df = pd.read_csv('Consumer_Complaints.csv',sep=",",nrows=50000,low_memory=False,)
#df = pd.read_csv('Consumer_Complaintsdemo.csv',sep=",",nrows=500)

#codificamos y limpiamos datos

col = ['Product', 'Consumer complaint narrative']
df = df[col]
df = df[pd.notnull(df['Consumer complaint narrative'])]
df.columns = ['Product', 'Consumer_complaint_narrative']
df['category_id'] = df['Product'].factorize()[0]
category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')

category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Product']].values)
print ("category_to_id",category_to_id)

#Vemos como se comportan

fig = plt.figure(figsize=(9,9))
df.groupby('Product').Consumer_complaint_narrative.count().plot.bar(ylim=0)
plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=10, norm='l2', encoding='latin-1', ngram_range=(1, 2),lowercase=True, stop_words="english")
features = tfidf.fit_transform(df.Consumer_complaint_narrative).toarray()
labels = df.category_id

#Correlacion

N = 4
for Product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Product))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

#Model Selection

X_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

models = [
    RandomForestClassifier(n_estimators=800, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 2 
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  print (model_name,accuracies)


  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
  
  cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
  
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
cv_df.groupby('model_name').accuracy.mean()


#Entrenar y Predecir



#LogisticRegression

#params = {'random_state':range(30,150,500) }
#arams = {'random_state':range(20) }


clf = LinearSVC()

model_grid = GridSearchCV(clf,{}, cv=CV, n_jobs=-1)
model_grid.fit(X_train_tfidf, y_train)
print("score de ",model_name,":",model_grid.score(X_train_tfidf, y_train))

print('LinearSVC:Precisión en el set de Entrenamiento: {:.2f}' .format(model_grid.score(X_train_tfidf, y_train)))
print('LinearSVC:Precisión en el set de Test:  {:.2f}' .format(model_grid.score(X_train_tfidf, y_train)))
joblib.dump(clf,"entrenamiento_LogisticRegression.joblib")


print(text1,model_grid.predict(count_vect.transform([text1])))
print(text2,model_grid.predict(count_vect.transform([text2])))
print(text3,model_grid.predict(count_vect.transform([text3])))
print(text4,model_grid.predict(count_vect.transform([text4])))
print(text5,model_grid.predict(count_vect.transform([text5])))


model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))



heatmap=sns.heatmap(conf_mat, annot=True, fmt='d')

#heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
#heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
#print('categories',category_id_df)


y = heatmap.yaxis.set_ticklabels(category_id_df.Product.values, rotation=0, ha='right', fontsize=8)
x = heatmap.xaxis.set_ticklabels(category_id_df.Product.values, rotation=25, ha='right', fontsize=8)


#sns.heatmap(conf_mat, annot=True, fmt='d',
#            xticklabels=category_id_df.Product.values, yticklabels=category_id_df.Product.values)
plt.title('Y Actual, X predicted')

plt.show()