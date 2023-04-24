from anime_processing import *
from anime_prompter import *
from extra import *
from knowledgebase import *
from ontology import *



print('Loading data...')
anime_data = pd.read_csv('./archive/anime.csv')
rating_data = pd.read_csv('./archive/rating.csv')

print('Processing data....')
anime_data = anime_data_processing(anime_data)

anime_index = anime_index_extraction(anime_data)
genres_list = genre_extraction(anime_data)

anime_pivot = anime_pivot_processing(anime_data, rating_data)
del rating_data


while True:
    print('Enter 1 if you want to be recommended anime with Collaborative Filtering ')
    print('and Content based filtering by entering the name of an anime you know')

    print('Enter 2 if you want an anime recommendation using content base filtering with a prevision of ranking')

    print('Enter 3 if you want an anime recommendation through an unsupervised model')

    print('Knowledge base function:')
    print('Enter 4 if you want anime details')
    print('Enter 5 if you want to check that an anime has a certain genre')
    print('Enter 6 if you want to list the genres of a certain anime')
    print('Enter 7 if you want to know if 2 anime have common genres (at least one)')
    print('Enter 8 if you want navigate between multiple genre and relative anime')
    print('Enter 9 if you want to know if one anime can you like it')

    print('Ontology function:')
    print('Enter 10 if you want explore the ontology')

    print('Extra function:')
    print('Enter 11 if you want to be recommended an anime by entering the genres you prefer the most')
    print('Enter 12 if you want if you want to know if an anime has been successful')
    print('Enter 13 if you want to know whether to run option 12 on a test model')


    print('Enter 0 if you want close the program')

    print('If you enter something else this information will be repeat')
    print('\n')
    recommended_choose = input('Enter your choice : ')

    if recommended_choose == '1':
        reccanimename(anime_data,anime_pivot,genres_list,anime_index)
    elif recommended_choose == '2':
        mainrecommender(anime_data,genres_list,anime_index)
    elif recommended_choose == '3':
        animesuggestion(anime_data)
    elif recommended_choose == '4':
        animedetails(anime_data)
    elif recommended_choose == '5':
        checkanime1genre(anime_data)
    elif recommended_choose == '6':
        checkanimegenre(anime_data)
    elif recommended_choose == '7':
        comparisonanime(anime_data)
    elif recommended_choose == '8':
        navigate(anime_data)
    elif recommended_choose == '9':
        likinganime(anime_data)
    elif recommended_choose == '10':
        ontology()
    elif recommended_choose == '11':
        reccanimegenres(anime_data)
    elif recommended_choose == '12':
        animesuccess(anime_data)
    elif recommended_choose == '13':
        animesuccesstest(anime_data)
        
    elif recommended_choose == '0':
        print('Closing program')
        break
