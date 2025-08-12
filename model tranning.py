import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib.testing import param
from numpy.f2py.auxfuncs import replace
from pandas import DataFrame
from sympy.matrices.expressions.matadd import combine
from torch.nn.functional import linear
plt.style.use("ggplot")
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
from nltk import sent_tokenize, word_tokenize, edge_closure
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,make_scorer
from time import time
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


df = pd.read_csv(r"C:\Users\saqla\Downloads\badwords.csv")
print(df.head())
print(df['label'].unique())

def perfrom_data_manipulation():
    df = pd.read_csv(r"C:\Users\saqla\Downloads\badwords.csv")

    for inedx in df.index:
        if df.loc[inedx,'label']==-1:
            df.loc[inedx,'label']=1
    return df
df = perfrom_data_manipulation()
print(df.head())
df['label'].unique()


def performdatadistribution(df):
    total = df.shape[0]
    num_non_toxic = df[df['label']==0].shape[0]
    slices = [num_non_toxic/total,(total-num_non_toxic)/total]#formula
    labeling = ['Non_Toxic','Toxic']
    explode = [0.2,0]
    plt.pie(slices,explode=explode,shadow=True,autopct='%1.1f%%',labels=labeling)
    plt.title('Number of Toxic Vs Non_Toxic Test Sample')
    plt.tight_layout()
    plt.show()

performdatadistribution(df)

def remove_pattern(input_txt,pattern):
    if (type(input_txt) == str):
        r = re.findall(pattern,input_txt)
        for i in r:
            input_txt = re.sub(re.escape(i),'',input_txt)
        return input_txt
    else:
        return ''

def datasetCleaning(df):
    df['length_headline']= df['headline'].str.len()
    combined_df = pd.concat([df,df],ignore_index=True) # Using pd.concat instead of append

    combined_df['tidy_tweet'] = np.vectorize(remove_pattern)(combined_df['headline'],'@[\w]*') # Corrected np.vectorize call

    combined_df['tidy_tweet'] = combined_df['tidy_tweet'].str.replace('[^a-zA-Z#]',' ', regex=True) # Added regex=True

    combined_df['tidy_tweet'] = combined_df['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

    combined_df['lenght_tidy_tweet'] = combined_df['tidy_tweet'].str.len()

    tokenized_tweet = combined_df['tidy_tweet'].apply(lambda x : x.split())

    nltk.download('wordnet')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokenized_tweet = tokenized_tweet.apply(lambda x :[lemmatizer.lemmatize(i) for i in x] ) # Assigned back to tokenized_tweet
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    combined_df['tidy_tweet'] = tokenized_tweet # Assign the processed tokens back to the DataFrame
    return combined_df,df


combined_df,df = datasetCleaning(df)

def performdatasplit(x,y,combined_df,df):
    X_train,X_test,Y_train,Y_test = train_test_split(combined_df['tidy_tweet'],combined_df['label'],test_size=x,random_state=y)
    print(f'Number of rows in the Total dataset {combined_df.shape[0]} ')
    print(f'Number of rows in the Tranin dataset {X_train.shape[0]} ')
    print(f'Number of rows in the Test dataset {X_test.shape[0]} ')
    file = open(r"C:\Users\saqla\Downloads\stopwords.txt",'r')
    content = file.read()
    content_list = content.split('\n')
    file.close()
    tfidvector = TfidfVectorizer(stop_words=content_list,lowercase=True)

    traning_data = tfidvector.fit_transform(X_train.values.astype('U'))
    testing_data = tfidvector.transform(X_test.values.astype('U'))
    file_name = 'tfidfVectorizer.pkl'
    pickle.dump(tfidvector.vocabulary_,open(file_name,'wb'))
    return X_train,X_test,Y_train,Y_test,testing_data,file_name,traning_data,content_list

X_train,X_test,Y_train,Y_test,testing_data,file_name,traning_data,content_list = performdatasplit(0.2,43,combined_df,df)

def pipeline(X_train,Y_train,X_test,Y_test):
    MODELS = [
        LinearSVC(),
        LogisticRegression(),
        MultinomialNB(),
        DecisionTreeClassifier(),
        AdaBoostClassifier(),
        BaggingClassifier(),
        SGDClassifier()
    ]
    size = len(Y_train)
    results = {}
    final_results = []
    for model in MODELS:
        results['Algorithm'] = model.__class__.__name__
        start = time()
        print(f'Traning Time : {model.__class__.__name__}')
        model.fit(X_train,Y_train)
        end = time()

        filename=model.__class__.__name__+".pkl"
        pickle.dump(model,open(filename,'wb'))

        results['Traning Time'] = end - start

        start = time()
        prediction_test = model.predict(X_test)
        prediction_train = model.predict(X_train)
        end = time()

        results['Prediction Time'] = end - start

        results['Accuracy : Test'] = accuracy_score(Y_test,prediction_test)
        results['Accuracy : Train'] = accuracy_score(Y_train ,prediction_train)

        results['F1 Score : Test'] = f1_score(Y_test,prediction_test)
        results['F1 Score : Train'] = f1_score(Y_train ,prediction_train)

        results['Precision : Test'] = precision_score(Y_test,prediction_test)
        results['Precision : Train'] = precision_score(Y_train ,prediction_train)

        results['Recall : Test'] = recall_score(Y_test,prediction_test)
        results['Recall : Train'] = recall_score(Y_train ,prediction_train)

        print(f"Traning {model.__class__.__name__} finished in {results['Traning Time']} sec")
        final_results.append(results.copy())
    return final_results

final_result = pipeline(traning_data,Y_train,testing_data,Y_test)

def perfromfinalresult(final_result):
    results = pd.DataFrame(final_result)
    results = results.reindex(columns=['Algorithm','Accuracy : Test','Precision : Test','Recall : Test',
                             'F1 Score : Test','Accuracy : Train','Prediction Time',
                             'Recall : Train','F1 Score : Train','Traning Time'])
    results = results.sort_values(by= 'F1 Score : Test',ascending=False)
    return results

results = perfromfinalresult(final_result)
results.reset_index(drop=True)

print(results.describe().loc[['min','max']])

best_acc = results[results['Accuracy : Test']  == results['Accuracy : Test'].max()]
best_f1 = results[results['F1 Score : Test'] == results['F1 Score : Test'].max()]
best_precidion = results[results['Precision : Test'] == results['Precision : Test'].max()]
best_recall = results[results['Recall : Train'] == results['Recall : Train'].max()]

sns.set_style('darkgrid')
plt.figure(figsize=(15,6))

barWidth = 0.17

bar1 = results['Accuracy : Test']
bar2 = results['F1 Score : Test']

r1 = np.arange(len(bar1))
r2 = [x+barWidth for x in r1]

pal = sns.color_palette()
plt.bar(r1,bar1,color=pal[0],width=barWidth,edgecolor='white',label='Test Accuracy')
plt.bar(r2,bar2,color=pal[1],width=barWidth,edgecolor='white',label='Test F1 Score')

plt.xlabel('Algorithm',fontweight='bold',fontsize=13)
plt.ylabel('Score',fontweight='bold',fontsize=13)
plt.xticks([r+barWidth for r in range(len(bar1))],results['Algorithm'],rotation=15,fontsize=11)

plt.legend(fontsize=13)

textstr = '\n'.join(['Best Accuracy: {:.3f}-{}'.format(best_acc['Accuracy : Test'].values[0],best_acc['Algorithm'].values[0]),
                     'Best F1 Score: {:.3f}-{}'.format(best_f1['F1 Score : Test'].values[0],best_f1['Algorithm'].values[0])])

props = dict(boxstyle='round',facecolor='lightgrey',alpha=0.5)
plt.title('Classification Summary of Algorithms',fontweight='bold',fontsize=17)
plt.show()

# First, fix the column name if needed
results.rename(columns={'Traning Time': 'Training Time'}, inplace=True)

# Remove rows where training time is missing
results_filtered = results.dropna(subset=['Training Time'])

# Calculate best/worst training and prediction times
best_train_time = results_filtered[results_filtered['Training Time'] == results_filtered['Training Time'].min()]
worst_train_time = results_filtered[results_filtered['Training Time'] == results_filtered['Training Time'].max()]
best_prediction_time = results[results['Prediction Time'] == results['Prediction Time'].min()]
worst_prediction_time = results[results['Prediction Time'] == results['Prediction Time'].max()]

# Plotting
plt.figure(figsize=(12,7))

barWidth = 0.17
bar1 = results['Training Time']
bar2 = results['Prediction Time']

r1 = np.arange(len(bar1))
r2 = [x+barWidth for x in r1]

plt.bar(r1, bar1, color=pal[0], width=barWidth, edgecolor='white', label='Training Time')
plt.bar(r2, bar2, color=pal[1], width=barWidth, edgecolor='white', label='Prediction Time')

plt.xlabel('Algorithm', fontweight='bold', fontsize=13)
plt.ylabel('Time (seconds)', fontweight='bold', fontsize=13)
plt.xticks([r+barWidth/2 for r in r1], results['Algorithm'], rotation=15, fontsize=11)

# Add summary box
textstr = '\n'.join([
    'Best Training Time: {:.3f} - {}'.format(best_train_time['Training Time'].values[0], best_train_time['Algorithm'].values[0]),
    'Worst Training Time: {:.3f} - {}'.format(worst_train_time['Training Time'].values[0], worst_train_time['Algorithm'].values[0]),
    'Best Prediction Time: {:.3f} - {}'.format(best_prediction_time['Prediction Time'].values[0], best_prediction_time['Algorithm'].values[0]),
    'Worst Prediction Time: {:.3f} - {}'.format(worst_prediction_time['Prediction Time'].values[0], worst_prediction_time['Algorithm'].values[0])
])
props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
plt.text(3.2, 12, textstr, fontsize=14, bbox=props)

plt.legend(fontsize=13)
plt.title('Training and Prediction Time of Algorithms', fontweight='bold', fontsize=17)
plt.tight_layout()
plt.show()

df.head()

data = ['You are so beautiful']
tfidf_vector = TfidfVectorizer(stop_words=content_list, lowercase=True, vocabulary=pickle.load(open("tfidfVectorizer.pkl","rb")))
preprocessed_data = tfidf_vector.fit_transform(data)

trained_model = pickle.load(open('LinearSVC.pkl', 'rb'))
print(trained_model.predict(preprocessed_data))

if(trained_model.predict(preprocessed_data)==1):
    print("Bulling")
else:
    print("Non-Bulling")

def tuning(clf,param_dict,X_train,Y_train,X_test,Y_test):
    scorer = make_scorer(f1_score)

    grid_obj = GridSearchCV(estimator=clf,param_grid=param_dict,scoring=scorer,cv=5)

    grid_fit = grid_obj.fit(X_train,Y_train)

    best_clf = grid_fit.best_estimator_

    perdiction = (clf.fit(X_train,Y_train)).predict(X_test)

    best_prediction = best_clf.predict(X_test)

    print(clf.__class__.__name__)
    print(f"Best Parameter : {grid_fit.best_params_}")
    print(f"Accuracy : {accuracy_score(Y_test,best_prediction)}")
    print(f"F1 score : {f1_score(Y_test,best_prediction)}")
    print(f"Precision : {precision_score(Y_test,best_prediction)}")
    print(f"Recall : {recall_score(Y_test,best_prediction)}")


param_grid = {
    'C':[0.25,0.5,0.75,1,1.2]
}
clf_model = LinearSVC()

tuning(clf_model,param_grid,traning_data,Y_train,testing_data,Y_test)

pickle.dump(clf_model,open("LinearSVC.pkl","wb"))
