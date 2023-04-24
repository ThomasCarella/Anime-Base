import pandas as pd

#Function that print an anime details
def checkanimegenre(anime_data):
    animename = input('Enter anime name : ')

    if animename in anime_data['name'].values:
        features = anime_data.columns[7:-1]
        print('Genre : ')
        for i in range(len(features.to_list())):
            if anime_data.at[ anime_data[anime_data['name'] == animename].index[0] ,features.to_list()[i]] == 1:
                print(features.to_list()[i])
        print('')
    else:
        print('There is no anime with that name in the dataset')

#Function that check one anime genre
def checkanime1genre(anime_data):
    animename = input('Enter anime name : ')
    if animename in anime_data['name'].values:
        features = anime_data.columns[7:-1]
        while True:
            print('Select one genres')
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
            print(' if you enter a number that does not match a listed gender, the question will be repeated')
            choice = input('Enter your choice : ')
            if int(choice) > 0 and int(choice) < 44:
                break
        if anime_data.at[ anime_data[anime_data['name'] == animename].index[0] ,features.to_list()[int(choice)]] == 1:
            print('Yes\n')
        else:
            print('No\n')
    else:
        print('There is no anime with that name in the dataset\n')

#Function that print yes if 2 anime have common genres (at least one)
def comparisonanime(anime_data):
    features = anime_data.columns[7:-1]
    exit = False
    animename1 = input('Enter first anime name : ')
    if animename1 in anime_data['name'].values:
        animename2 = input('Enter second anime name : ')
        if animename2 in anime_data['name'].values:
            for i in range(len(features.to_list())):
                for j in range(len(features.to_list())):
                    if anime_data.at[ anime_data[anime_data['name'] == animename1].index[0] ,features.to_list()[i]] == 1:
                        if anime_data.at[ anime_data[anime_data['name'] == animename1].index[0] ,features.to_list()[i]] == anime_data.at[ anime_data[anime_data['name'] == animename2].index[0] ,features.to_list()[j]]:
                            print('Yes\n')
                            exit = True
                            break
                if exit == True:
                    break  
        else:
            print('There is no anime with that name in the dataset\n')
    else:
        print('There is no anime with that name in the dataset\n')

    if exit == False:
        print('No\n')

#Function that print an anime details
def animedetails(anime_data):
    animename = input('Enter anime name as example Naruto : ')
    if animename in anime_data['name'].values:
        print(pd.DataFrame({'Anime name': anime_data['name'].loc[anime_data['name'] == animename].values,
                            'Rating': anime_data['rating'].loc[anime_data['name'] == animename].values,
                            'Type': anime_data['type'].loc[anime_data['name'] == animename].values}))

    else:
        print('There is no anime with that name in the dataset\n')

def navAnime(filtered_anime):
  features = filtered_anime.columns[7:-1].to_list()
  meanRating = []
  numAnime = []
  i = 0
  while i < len(features):
    if filtered_anime[filtered_anime[features[i]]==1].shape[0] > 0:
      numAnime.append(filtered_anime[filtered_anime[features[i]]==1].shape[0])
      temp_df = filtered_anime[filtered_anime[features[i]]==1]
      meanRating.append(temp_df['rating'].mean())
      del temp_df
      i = i + 1
    else:
      features.pop(i)
  print(pd.DataFrame({'Anime Genres': features,'Number of anime': numAnime}))

def printAnimedetails(filtered_anime):
    print(pd.DataFrame({'Name': filtered_anime['name'], 'Genres': filtered_anime['genre'],'Type': filtered_anime['type'],'rating': filtered_anime['rating']}))

#Function for navigate between multiple genre and relative anime
def navigate(anime_data):
  filtered_anime = anime_data.copy()
  print('All genres are:')
  print('\n'.join(filtered_anime.columns[7:-1]))
  print('All Type are:')
  print('\n'.join(list(set(filtered_anime.type.values))))
  print('If you want stop navigate insert STOP')
  print('If you want see the relative anime insert DETAILS')
  while True:
    choice = input('Insert anime genre/type : ')
    if choice == 'STOP':
      break
    elif choice in filtered_anime.columns[7:-1].to_list():
      filtered_anime = filtered_anime[filtered_anime[choice]==1].copy()
      navAnime(filtered_anime)
    elif choice in filtered_anime.type.values:
      filtered_anime = filtered_anime[filtered_anime['type']==choice].copy()
      navAnime(filtered_anime)
    elif choice == 'DETAILS':
      printAnimedetails(filtered_anime)
    else:
      print('There is not a genre/type with this name')
      print('Warning : There is only 1 type for anime')
  del filtered_anime
#given a name of an anime that is known, and a name of an anime, it returns the printout of the probability of liking
def likinganime(anime_data):
    features = anime_data.columns[7:-1]
    animename1 = input('Enter first anime name(that you like it) : ')
    if animename1 in anime_data['name'].values:
      count0 = 0
      animename2 = input('Enter second anime name(that you want know if you can like it) : ')
      if animename2 in anime_data['name'].values:
        count1 = 0
        for i in range(len(features.to_list())):
          if anime_data.at[ anime_data[anime_data['name'] == animename1].index[0] ,features.to_list()[i]] == 1:
            count0 = count0 + 1
            if anime_data.at[ anime_data[anime_data['name'] == animename2].index[0] ,features.to_list()[i]] == 1:
              count1 = count1 + 1
      else:
        print('There is no anime with that name in the dataset\n')
        return
    else:
      print('There is no anime with that name in the dataset\n')
      return
    if count0 is not 0:
        print("{0:.0%}".format(count1/count0))
        print('\n')
    else:
      print('There is no Compatibily')
