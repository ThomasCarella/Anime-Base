import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import time



# Function that print the most similar anime using
# Random Forest Classifier ,  Support Vector Classifier , 
# Bagging classifier using k-nearest neighbors vote and
# Bagging classifier using Decision Tree Classifier
def anime_genre_like(anime_select,test):
    #column of features
    features = anime_select.columns[7:-1]
    #column of anime id
    y = anime_select['anime_id']
    print('Processing data...')
    time_sec =time.time()
    clf = RandomForestClassifier(n_jobs=10, random_state=42, max_depth=10)
    clf.fit(anime_select[features], y)
    print('Random Forest Classifier recommendation: ')
    print('Warning : the maximum depth of the tree is limited to 10 in this particular test\n')
    print(pd.DataFrame({'Anime name' : anime_select['name'].loc[anime_select['anime_id'] == clf.predict(test[features])[0]]}))
    print('Time to calculate it : {:0.2f} seconds'.format(time.time()-time_sec))
    print('\n')
    del clf

    time_sec =time.time()
    svc = SVC()
    print('Processing data...')
    svc.fit(anime_select[features], y)
    print('Support Vector Classifier recommendation: \n')
    print(pd.DataFrame({'Anime name' : anime_select['name'].loc[anime_select['anime_id'] == svc.predict(test[features])[0]]}))
    print('Time to calculate it : {:0.2f} seconds'.format(time.time()-time_sec))
    print('\n')
    del svc

    time_sec =time.time()  
    model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
    print('Processing data...')
    model.fit(anime_select[features],y)
    print('Bagging classifier using k-nearest neighbors vote recommendation: \n')
    print(pd.DataFrame({'Anime name' : anime_select['name'].loc[anime_select['anime_id'] == model.predict(test[features])[0]]}))
    print('Time to calculate it : {:0.2f} seconds'.format(time.time()-time_sec))
    print('\n')
    del model

    time_sec =time.time()
    model=BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=10),random_state=0,n_estimators=100)
    print('Processing data...')
    model.fit(anime_select[features],y)
    print('Bagging classifier using Decision Tree Classifier recommendation: ')
    print('Warning : the maximum depth of the tree is limited to 10 in this particular test\n')
    print(pd.DataFrame({'Anime name' : anime_select['name'].loc[anime_select['anime_id'] == model.predict(test[features])[0]]}))
    print('Time to calculate it : {:0.2f} seconds'.format(time.time()-time_sec))
    print('\n')
    del model

    del anime_select

# Function that recommended an anime by entering the genres you prefer the most
def reccanimegenres(anime_data):
    print('Warning : to reduce the use of ram, the database size will be reduced')
    print('based on the type of anime such as tv movies etc ..')
    while True:
        print('\nEnter 1 for seach with type Movie')
        print('Enter 2 for seach with type Music')
        print('Enter 3 for seach with type ONA')
        print('Enter 4 for seach with type OVA')
        print('Enter 5 for seach with type Special')
        print('Enter 6 for seach with type TV')
        type_select = input('Enter your choice : ')
        if type_select == '1':
            anime_select = anime_data.loc[anime_data['type'] == 'Movie']
            break
        elif type_select == '2':
            anime_select = anime_data.loc[anime_data['type'] == 'Music']
            break
        elif type_select == '3':
            anime_select = anime_data.loc[anime_data['type'] == 'ONA']
            break
        elif type_select == '4':
            anime_select = anime_data.loc[anime_data['type'] == 'OVA']
            break
        elif type_select == '5':
            anime_select = anime_data.loc[anime_data['type'] == 'Special']
            break
        elif type_select == '6':
            anime_select = anime_data.loc[anime_data['type'] == 'TV']
            break

    test = anime_select[0:0]
    test.loc[len(test.index)] = [0,'0','0','0',0,0,0,  
                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]
    features = anime_data.columns[7:-1]

    print('Select one or more genres')
    print(' 1 Action\n 2 Adventure\n 3 Cars\n 4 Comedy\n 5 Dementia')
    print(' 6 Demons\n 7 Drama\n 8 Ecchi\n 9 Fantasy')
    print(' 10 Game\n 11 Harem\n 12 Hentai\n 13 Historical\n 14 Horror')
    print(' 15 Josei\n 16 Kids\n 17 Magic\n 18 Martial Arts')
    print(' 19 Mecha\n 20 Military\n 21 Music\n 22 Mystery\n 23 Parody')
    print(' 24 Police\n 25 Psychological\n 26 Romance\n 27 Samurai')
    print(' 28 School\n 29 Sci-Fi\n 30 Seinen\n 31 Shoujo\n 32 Shoujo Ai')
    print(' 33 Shounen\n 34 Shounen Ai\n 35 Slice of Life')
    print(' 36 Space\n 37 Sports\n 38 Super Power\n 39 Supernatural')
    print(' 40 Thriller\n 41 Vampire\n 42 Yaoi\n 43 Yuri')
    print(' 44 test with genres of Shingeki no Kyojin a.k.a. Attack of titan')
    print('Enter 0 to stop selection, enter 45 if you do not want this type of recommendation')

    while True:
        choice = input('Enter your choice : ')
        if int(choice) > 0 and int(choice) < 44:
            test.at[0,features.to_list()[int(choice)]] = 1

            print('Genre acquired!')
            print('Enter 0 to stop selection, enter 45 if you do not want this type of recommendation')            
        elif  choice == '44':
            test = test[0:0]
            test.loc[len(test.index)] = [0,'0','0','0',0,0,0,1,0,0,0,0,0,1,0,1,
                                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0
                                ,0,1,0,0,0,0,0,0 ]
            anime_genre_like(anime_select,test)
            break
            
        elif  choice == '0':
            anime_genre_like(anime_select,test)        
            break

        elif  choice == '45':
            print('Close...')
            break
        else:
            print('The entered input is invalid, please enter a correct one ...')



#Function to know if an anime has been successful
def animesuccess(anime_data):
    features = anime_data.columns[7:-1]
    df = anime_data
    df = df.drop('name',axis=1)
    df = df.drop('type',axis=1)
    df = df.drop('episodes',axis=1)
    df = df.drop('genre',axis=1)

    y = df['success']
    x = df.drop('success',axis= 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, shuffle=True, random_state=1)   
    #Standardizzazione dei valori di X
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)
     
    animename = input('Enter anime name as example Gintama : ')
    if animename in anime_data['name'].values:
        test = anime_data.loc[anime_data['name'] == animename]
        test = test.drop('name',axis=1)
        test = test.drop('type',axis=1)
        test = test.drop('episodes',axis=1)
        test = test.drop('success',axis= 1)
        test = test.drop('genre',axis= 1)
        print('For Gaussian Naive Bayes Classifier : ')
        model = GaussianNB()
        model.fit(x_train,y_train)
        pred = model.predict(test)
        pred = np.array(pred).astype(int)
        if pred[0] == 1:
            print('Yes')
        else:
            print('No')

        print('For Logistic Regression : ')
        model = LogisticRegression()
        model.fit(x_train,y_train)
        pred = model.predict(test)
        pred = np.array(pred).astype(int)
        if pred[0] == 1:
            print('Yes')
        else:
            print('No')
    
        print('For KNeighbors Classifier : ')
        model = KNeighborsClassifier()
        model.fit(x_train,y_train)
        pred = model.predict(test)
        pred = np.array(pred).astype(int)
        if pred[0] == 1:
            print('Yes')
        else:
            print('No')

        print('For Decision Tree Classifier : ')
        model = DecisionTreeClassifier()
        model.fit(x_train,y_train)
        pred = model.predict(test)
        pred = np.array(pred).astype(int)
        if pred[0] == 1:
            print('Yes')
        else:
            print('No')

        print('For Neural Network : ')
        model = MLPClassifier()
        model.fit(x_train,y_train)
        pred = model.predict(test)
        pred = np.array(pred).astype(int)
        if pred[0] == 1:
            print('Yes')
        else:
            print('No')

        print('For Random Forest Classifier : ')
        model = RandomForestClassifier()
        model.fit(x_train,y_train)
        pred = model.predict(test)
        pred = np.array(pred).astype(int)
        if pred[0] == 1:
            print('Yes')
        else:
            print('No')

        print('For Gradient Boosting Classifier : ')
        model = GradientBoostingClassifier()
        model.fit(x_train,y_train)
        pred = model.predict(test)
        pred = np.array(pred).astype(int)
        if pred[0] == 1:
            print('Yes')
        else:
            print('No')
        print('')
    else:
        print('There is no anime with that name in the dataset\n')

#Function to know if a test model has been successful
def animesuccesstest(anime_data):
    features = anime_data.columns[7:-1]
    df = anime_data
    df = df.drop('name',axis=1)
    df = df.drop('type',axis=1)
    df = df.drop('episodes',axis=1)
    df = df.drop('genre',axis=1)

    y = df['success']
    x = df.drop('success',axis= 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, shuffle=True, random_state=1)   
    #Standardizzazione dei valori di X
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)
    print('1 mean Success')
    print('0 means Flop\n')

    print('For Gaussian Naive Bayes Classifier : ')
    model = GaussianNB()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    print(metrics.classification_report(y_test,pred))
    print('')

    print('For Logistic Regression : ')
    model = LogisticRegression()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    print(metrics.classification_report(y_test,pred))
    print('')
  
    print('For KNeighbors Classifier : ')
    model = KNeighborsClassifier()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    print(metrics.classification_report(y_test,pred))
    print('')

    print('For Decision Tree Classifier : ')
    model = DecisionTreeClassifier()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    print(metrics.classification_report(y_test,pred))
    print('')

    print('For Neural Network : ')
    model = MLPClassifier()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    print(metrics.classification_report(y_test,pred))
    print('')

    print('For Random Forest Classifier : ')
    model = RandomForestClassifier()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    print(metrics.classification_report(y_test,pred))
    print('')

    print('For Gradient Boosting Classifier : ')
    model = GradientBoostingClassifier()
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    print(metrics.classification_report(y_test,pred))
    print('')
