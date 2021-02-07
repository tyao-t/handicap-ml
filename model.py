import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.manifold import Isomap
from keras.models import Sequential
from keras.layers import Dense


def load_data():
    dfs = []
    for filename in os.listdir('CSVs/'):
        year_df = pd.read_csv('CSVs/'+filename)
        year_df['winner'] = 1*(year_df['rank'] == 1)
        dfs.append(year_df)
    df = pd.concat(dfs)
    df.index = range(0,df.shape[0])
    return df

def process_data(df):

    # defining country mapping
    country_to_id_mapping = {}
    id_to_country_mapping = {}

    id_count = 0
    for code in df['code'].unique():
        country_to_id_mapping[code] = id_count
        id_to_country_mapping[id_count] = code

        id_count += 1

    # new data
    new_ids = []
    for code in df['code']:
        new_ids.append(country_to_id_mapping[code])
    df['country_id'] = new_ids

    df['top_5'] = (df['rank'] < 5)*1
    df['top_10'] = (df['rank'] < 10)*1
    df['top_15'] = (df['rank'] < 15)*1
    df['top_25'] = (df['rank'] < 25)*1
    df['top_50'] = (df['rank'] < 50)*1
    df['top_100'] = (df['rank'] < 100)*1

    # dropping columns
    df = df.drop(columns = ['problem1','problem2','problem3','problem4','problem5','problem6'])

    df['next_year_winner'] = 0
    new_dfs = []
    for year in df['year'].unique():
        if year < 2020:
            tmp1 = df[df['year'] == year + 1].copy()
            next_year_winner = tmp1[tmp1['rank'] == 1]['country_id'].iloc[0]
            new_df = df[df['year'] == year].copy()
            new_df['next_year_winner'] = (new_df['country_id'] == next_year_winner)*1

            for i in [5,10,15,25,50,100]:
                top_i = tmp1[tmp1['top_'+str(i)] == 1]
                countries = list(top_i['country_id'].unique())
                new_df['next_year_top_'+str(i)] = new_df['country_id'].isin(countries)*1


            new_dfs.append(new_df)
        else:
            new_df = df[df['year'] == 2020].copy()
            new_df['next_year_winner'] = 0
            new_dfs.append(new_df)

    df = pd.concat(new_dfs)

    dfs = []
    for code in df['code'].unique():
        code_df = df[df['code'] == code].sort_values(by = ['year'])
        code_df['total_wins'] = code_df['winner'].cumsum()
        code_df['total_gold'] = code_df['gold_medals'].cumsum()
        code_df['total_silver'] = code_df['silver_medals'].cumsum()
        code_df['total_bronze'] = code_df['bronze_medals'].cumsum()
        code_df['min_rank'] = code_df['rank'].cummin()
        code_df['max_rank'] = code_df['rank'].cummax()
        code_df['average_rank'] = code_df['rank'].cumsum()/np.arange(1,code_df.shape[0]+1)
        for i in [2,3,4]:
            code_df['average_rank_'+str(i)] = code_df['rank'].rolling(window = i).mean()
            code_df['recent_wins_'+str(i)] = code_df['rank'].rolling(window = i).sum()

        dfs.append(code_df)
    df = pd.concat(dfs).fillna(0)

    return df, country_to_id_mapping,id_to_country_mapping


df = load_data()
df, country_to_id_mapping,id_to_country_mapping = process_data(df)


features = ['total', 'rank','winner', 'country_id',
            'top_5','top_10', 'top_15', 'top_25', 'top_50', 'top_100',
            'total_wins', 'min_rank',
            'max_rank', 'average_rank', 'average_rank_2', 'recent_wins_2',
            'average_rank_3', 'recent_wins_3', 'average_rank_4', 'recent_wins_4']


target = 'next_year_winner'




train, test = df[df['year'] <= 2019],  df[df['year'] == 2020]
X_train, y_train = train[features], train[target].astype(int)
X_test = test[features]


model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150, batch_size=10)


pred_probs = model.predict(X_test)
country_ids = X_test['country_id']
for_df = []
for i,j in zip(pred_probs, country_ids):
    win_prob = i
    country = id_to_country_mapping[j]
    odds = win_prob/(1-win_prob)
    for_df.append({'country':country,'prob':win_prob,'odds':odds})
results = pd.DataFrame(for_df)
results.to_csv('winning_odds.csv')


from sklearn.model_selection import GridSearchCV
parameters = {'activation':["logistic", "relu"],
             'solver' : ['lbfgs','sgd','adam'],
             'alpha' : [0.00001,0.0001,0.001,0.01]}
nn = MLPClassifier(random_state = 42, max_iter = 1000)
clf = GridSearchCV(nn, parameters)
clf.fit(X_train, y_train)
optimal_params = clf.best_params_


model = MLPClassifier(**optimal_params)
model.fit(X_train, y_train)
model.score(X_train, y_train)
pred_probs = model.predict_proba(X_test)
country_ids = X_test['country_id']
for_df = []
for i,j in zip(pred_probs, country_ids):
    win_prob = i[1]
    country = id_to_country_mapping[j]
    odds = win_prob/(1-win_prob)
    for_df.append({'country':country,'prob':win_prob,'odds':odds})
results = pd.DataFrame(for_df)
results.to_csv('winning_odds_v2.csv')



nn_pipe = Pipeline([('pca',PCA()),('minmax',MinMaxScaler()),('nn', MLPClassifier())])
nn_pipe.fit(X_train, y_train)
pred_probs = nn_pipe.predict_proba(X_test)
country_ids = X_test['country_id']
for_df = []
for i,j in zip(pred_probs, country_ids):
    win_prob = i[1]
    country = id_to_country_mapping[j]
    odds = win_prob/(1-win_prob)
    for_df.append({'country':country,'prob':win_prob,'odds':odds})
results = pd.DataFrame(for_df)
results.sort_values(by = ['prob'], ascending = False).head(20)
nn_pipe.score(X_train, y_train)




for i in [5,10,15,25,50,100]:
    target = 'next_year_top_'+str(i)
    train, test = df[df['year'] <= 2019],  df[df['year'] == 2020]
    X_train, y_train = train[features], train[target].astype(int)
    X_test = test[features]

    model = MLPClassifier()
    model.fit(X_train, y_train)
    model.score(X_train, y_train)

    pred_probs = model.predict_proba(X_test)
    country_ids = X_test['country_id']
    for_df = []
    for i,j in zip(pred_probs, country_ids):
        win_prob = i[1]
        country = id_to_country_mapping[j]
        odds = win_prob/(1-win_prob)
        for_df.append({'country':country,'prob':win_prob,'odds':odds})
    results = pd.DataFrame(for_df)
    results.to_csv('odds_top_'+str(i))
