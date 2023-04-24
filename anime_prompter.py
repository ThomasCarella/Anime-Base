import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity   
import time

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold

from scipy.stats import pearsonr
import time


#function that carries out a recommendation of anime, using the Cosine similarity
# as a metric for the similarity calculation and obtaining the amount of time used for processing
def recommend_cosine(anime_data,genres_list,anime_index,title):
  time_sec =time.time()  
  # using the term frequency–inverse document frequency
  tf = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')  
  tf_matrix = tf.fit_transform(genres_list)

  cosine_sim = sigmoid_kernel(tf_matrix, tf_matrix)
  idx = anime_index[title]
  sim_scores = list(enumerate(cosine_sim[idx]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  # scores of the 10 most similar anime
  sim_scores = sim_scores[1:11]
  # anime indices
  anime_indices = [i[0] for i in sim_scores]
  # print the top 10 most similar anime
  print('Using Content based filtering, the Top 10 most similar anime are :\n')
  print(pd.DataFrame({'Anime name': anime_data['name'].iloc[anime_indices].values,
                                  'Rating': anime_data['rating'].iloc[anime_indices].values}))
  print('Time to calculate it : {:0.2f} seconds'.format(time.time()-time_sec))
  return anime_indices

#function that carries out a recommendation of anime, using the Euclidean distance
# as a metric for the similarity calculation and obtaining the amount of time used for processing
def recommend_euclidean(anime_data,genres_list,anime_index,title):
  time_sec =time.time()  
  # using the term frequency–inverse document frequency
  tf = TfidfVectorizer(analyzer='word')
  tf_matrix = tf.fit_transform(genres_list)
  euclidean = euclidean_distances(tf_matrix)
  idx = anime_index[title]
  sim_scores = list(enumerate(euclidean[idx]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  # scores of the 10 most similar anime
  sim_scores = sim_scores[1:11]
  # anime indices
  anime_indices = [i[0] for i in sim_scores]
  # print the top 10 most similar anime
  print('Using Content based filtering, the Top 10 most similar anime are :\n')
  print(pd.DataFrame({'Anime name': anime_data['name'].iloc[anime_indices].values,
                                  'Rating': anime_data['rating'].iloc[anime_indices].values}))
  print('Time to calculate it : {:0.2f} seconds'.format(time.time()-time_sec))
  return anime_indices

#function that carries out a recommendation of anime, using the Pearson correlation
# as a metric for the similarity calculation and obtaining the amount of time used for processing
def recommend_pearson(anime_data,genres_list,anime_index,title):
  time_sec =time.time()  
  # using the term frequency–inverse document frequency
  tf = TfidfVectorizer(analyzer='word')
  tf_matrix = tf.fit_transform(genres_list)
  tf_matrix = tf_matrix.toarray()
  idx = anime_index[title]
  pearson = []
  for i in range(len(tf_matrix)):
      pearson.append(pearsonr(tf_matrix[idx], tf_matrix[i])[0])
  sim_scores = list(enumerate(pearson))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  # anime indices
  anime_indices = [i[0] for i in sim_scores]
  # print the top 10 most similar anime
  print('Using Content based filtering, the Top 10 most similar anime are :\n')
  print(pd.DataFrame({'Anime name': anime_data['name'].iloc[anime_indices].values,
                                  'Rating': anime_data['rating'].iloc[anime_indices].values}))
  print('Time to calculate it : {:0.2f} seconds'.format(time.time()-time_sec))
  return anime_indices

#function that calculates the percentage of spasity that a dataset can have
def calculating_sparsity(data):
    
    data = data.to_numpy()

    sparsity = 1.0 - (np.count_nonzero(data) / float(data.size))

    print('Sparsity of the dataset:', sparsity*100, "%\n")


#function that performs a Randomized Search of the hyperparameters of the chosen model
def RandomizedSearch(hyperparameters, X_train, y_train):
    knn = KNeighborsClassifier()
    cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    randomSearch = RandomizedSearchCV(estimator=knn, cv=cvFold, param_distributions=hyperparameters)
    best_model = randomSearch.fit(X_train, y_train)

    return best_model

#function that evaluates a series of metrics on the model, such as accuracy, recall f1 etc...
def modelEvaluation(y_test, y_pred, pred_prob):
    print('Classification report: \n', classification_report(y_test, y_pred))
    #Checking performance our model with ROC Score.
    roc_score = roc_auc_score(y_test, pred_prob, multi_class='ovr')
    print('ROC score: ', roc_score)

    return roc_score

#function that searches for the absolute best hyperparameters, 
# repeating the function that performs the Randomized Search several times
#returns a dictionary containing hyperparameters and ROC score
def HyperparametersSearch(X_train, X_test, y_train, y_test):
    result = {}
    n_neighbors = list(range(1,30))
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'hamming']
    #Convert to dictionary
    hyperparameters = dict(metric=metric, weights=weights, n_neighbors=n_neighbors)
    i = 0
    while i < 15:
        best_model = RandomizedSearch(hyperparameters, X_train, y_train)
        bestweights = best_model.best_estimator_.get_params()['weights']
        bestMetric = best_model.best_estimator_.get_params()['metric']

        bestNeighbours = best_model.best_estimator_.get_params()['n_neighbors']
        knn = KNeighborsClassifier(n_neighbors=bestNeighbours, weights=bestweights, algorithm='auto', metric=bestMetric, metric_params=None, n_jobs=None)
        knn.fit(X_train,y_train)

        pred_prob = knn.predict_proba(X_test)
        roc_score = roc_auc_score(y_test, pred_prob, multi_class='ovr')

        result[i] = {'n_neighbors' : bestNeighbours, 'metric' : bestMetric, 'weights' : bestweights, 'roc_score' : roc_score}
        i = i + 1 

    result = dict(sorted(result.items(), key = lambda x: x[1]['roc_score'], reverse=True))
    first_el = list(result.keys())[0]
    result = list(result[first_el].values())
    return result

#function that looks for the best statistics to apply to the chosen model, gradually evaluating the performance
def SearchingBestModelStats(X_train, X_test, y_train, y_test):

    print('Initial model composition with basic hyperparameters')
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', p=2, metric='minkowski', metric_params=None, n_jobs=None)
    knn.fit(X_train,y_train)
    #show first 10 model predictions on the test data
    print('\nPredictions of the first 10 elements: ',knn.predict(X_test)[0:10],'\nActual values: ', y_test[0:10])

    y_pred = knn.predict(X_test)
    pred_prob = knn.predict_proba(X_test)

    print('Model evaluation...\n')
    modelEvaluation(y_test, y_pred, pred_prob)

    print('Accuracy is low, let"s try to improve the quality of the predictions\n')

    result = {}
    result = HyperparametersSearch(X_train, X_test, y_train, y_test)

    #Print the value of best Hyperparameters for randomizedsearch
    print('WITH GRID SEARCH:\n')

    bestweights = result[2]
    print('Best weights:', bestweights)

    bestMetric = result[1]
    print('Best metric:', bestMetric)

    bestNeighbours = result[0]
    print('Best n_neighbors:', bestNeighbours)

    #recomposition of the model with the new parameters and evaluation of the same
    print('Let"s recompose the model using the new hyperparameters...')

    knn = KNeighborsClassifier(n_neighbors=bestNeighbours, weights=bestweights, algorithm='auto', metric=bestMetric, metric_params=None, n_jobs=None)
    knn.fit(X_train,y_train)

    #show first 10 model predictions on the test data
    print('Predictions of the top 10 items in the ranking category: ',knn.predict(X_test)[0:10],'Actual values:', y_test[0:10])

    y_pred = knn.predict(X_test)
    pred_prob = knn.predict_proba(X_test)
    modelEvaluation(y_test, y_pred, pred_prob)

    print('We have increased the accuracy of our model')
    print('Now we can proceed to the recommendation phase...')

    return knn
#Recommendation function with the best model chosen
def mainrecommender(anime_data,genres_list,anime_index):

    print('Do you want proceed with a name test : Shingeki no Kyojin a.k.a. Attack of titan ?')
    title_test_choose = input('Enter y for a yes, something else for no : ')
    if title_test_choose == 'y':
        title = 'Shingeki no Kyojin'
        recommender(anime_data,title,genres_list,anime_index)
    else:
        while True:
            while True:
                title = input('Enter name of a anime and check if there are similar name : ')
                contain_title = anime_data[anime_data['name'].str.contains(title)]
                if(not contain_title.empty):
                    print(pd.DataFrame(contain_title.name))
                    break
                else:
                    print('There is no anime with this name')
            title = input('Enter name of a anime: ')
            if title in anime_data.values :
                recommender(anime_data,title,genres_list,anime_index)
                break
            else:
                print('The name of the anime you entered is missing or misspelled\n')
    
def recommender(anime_data,title,genres_list,anime_index):
    anime_data['ratingClone'] = anime_data.loc[:, 'rating']
    anime_data.ratingClone = anime_data.ratingClone.astype(int)
    anime_data.loc[anime_data["ratingClone"] < 6 , "ratingClone"] = 6
    anime_data.loc[anime_data["ratingClone"] > 9 , "ratingClone"] = 9
    df = anime_data
    df = df.drop('name',axis=1)
    df = df.drop('type',axis=1)
    df = df.drop('episodes',axis=1)
    df = df.drop('rating',axis=1)
    df = df.drop('anime_id',axis=1)
    df = df.drop('genre',axis=1)
    df = df.drop('members',axis=1)
    df = df.drop('success',axis=1)

    x = anime_data.iloc[:, 7:-2]
    y = anime_data['ratingClone'].values

    test = anime_data.loc[anime_data['name'] == title]
    test = test.drop('name',axis=1)
    test = test.drop('type',axis=1)
    test = test.drop('episodes',axis=1)
    test = test.drop('rating',axis=1)
    test = test.drop('anime_id',axis=1)
    test = test.drop('genre',axis=1)
    test = test.drop('members',axis=1)
    test = test.drop('success',axis=1)
    test = test.drop('ratingClone',axis= 1)
    an_index = recommend_pearson(anime_data,genres_list,anime_index,title)

    recommend_data = anime_data[['name','rating']].iloc[an_index]
    predict_data = anime_data[anime_data.columns[7:-2]].iloc[an_index]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1,stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    predict_data = scaler.transform(predict_data)
    knn = SearchingBestModelStats(X_train, X_test, y_train, y_test)

    knn.fit(X_train,y_train)

    pred = knn.predict(predict_data)
    recommend_data['pred'] = pred

    print('The 10 anime most similar to the one proposed with a prediction on the ranking: \n')
    i = 0
    while(i<9):
        print('Name : ',recommend_data.name[i])
        print('Rating : ',recommend_data.rating[i])
        print('Prediction Rating',recommend_data.pred[i],'\n')

        i = i + 1


#Function an anime recommendation through an unsupervised model
def animesuggestion(anime_data):
    animename = input('Enter anime name as example Gintama : ')
    if animename in anime_data['name'].values:
        test = anime_data.loc[anime_data['name'] == animename]    
        train = anime_data.drop(['name', 'type', 'episodes','rating','members','success','genre'], axis=1)
        test = test.drop(['name', 'type', 'episodes','rating','members','success','genre'], axis=1)

        model = KMeans(n_clusters=1000, random_state=0).fit(train)
        pred = model.predict(test)

        print('Similar anime using unsupervised learning : ')
        for i in range(len(anime_data)):
            if model.labels_[i] == pred[0]:
                print(anime_data['name'].iloc[i])
        print('')
    else:
        print('There is no anime with that name in the dataset\n')


#Function that print the Top 10 most similar anime using Collaborative Filtering and Content based filtering 
def anime_similarity(anime_pivot,anime_data,genres_list,anime_index, title):

    time_sec =time.time()
    anime_matrix = csr_matrix(anime_pivot.values)
    # using the k-nearest neighbors algorithm
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(anime_matrix)
    distances, indices = model_knn.kneighbors(anime_pivot.loc
                        [anime_pivot.index == title].values.reshape(1, -1), n_neighbors = 11)

    # print the top 10 most similar anime
    print('Using Collaborative Filtering, the Top 10 most similar anime are :')
    print('Warning : to reduce the use of ram,')
    print('the size of the database will be reduced, anime with at least 500 votes will be considered')
    for i in range(0, len(distances.flatten())):
        print('{0}: {1}, with distance of {2}:'
            .format(i, anime_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
    print('Time to calculate it : {:0.2f} seconds\n'.format(time.time()-time_sec))

    print('Using Cosine Similaryty')
    recommend_cosine(anime_data,genres_list,anime_index,title)

    print('Using Euclidean Distances')
    recommend_euclidean(anime_data,genres_list,anime_index,title)

    print('Using Pearson Similaryty')
    recommend_pearson(anime_data,genres_list,anime_index,title)
    
    calculating_sparsity(anime_data)


# Function that recommended anime with Collaborative Filtering 
# and Content based filtering by entering the name of an anime you know
def reccanimename(anime_data,anime_pivot,genres_list,anime_index):
    print('Do you want proceed with a name test : Shingeki no Kyojin a.k.a. Attack of titan ?')
    title_test_choose = input('Enter y for a yes, something else for no : ')
    if title_test_choose == 'y':
        title = 'Shingeki no Kyojin'
        anime_similarity(anime_pivot,anime_data,genres_list,anime_index,title)
    else:
        while True:
            while True:
                title = input('Enter name of a anime and check if there are similar name : ')
                contain_title = anime_pivot[anime_pivot.index.str.contains(title)]
                if(not contain_title.empty):
                    print(pd.DataFrame(contain_title.index))
                    break
                else:
                    print('There is no anime with this name')
            title = input('Enter name of a anime: ')
            if title in anime_pivot.index.values :
                anime_similarity(anime_pivot,anime_data,genres_list,anime_index,title)
                del title
                break
            else:
                print('The name of the anime you entered is missing or misspelled\n')

