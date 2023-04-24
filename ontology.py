from owlready2 import *

def ontology():

    print("\nWELCOME TO THE ONTOLOGY\n")
    while(True):
        print("Select what you would like to explore:\n\n1) Classes view\n2) Object properties view\n3) Data properties view\n4) Example query view\n5) Exit Ontology\n")

        risp_menù = input("Enter your choice here:\n")

        ontology = get_ontology('archive/anime.owl').load()

        if risp_menù == '1':
            print("\nclasses present in the ontology:\n")
            classes = list(ontology.classes())
            print(classes)

            while(True):
                print("\nWould you like to explore any particular class better?\n\n1) UserID\n2) Anime_Name\n3) Ranking\n4) Genre\n5) Type\n6) Vote\n")
                risposta_class = input("Enter your choice here:\n")

                if risposta_class == '1':
                    print("\nList of UserIDs present:\n")
                    agents = ontology.search(is_a = ontology.UserID)
                    print(agents)
                elif risposta_class == '2':
                    print("\nList of Anime_Names present:\n")
                    games = ontology.search(is_a = ontology.Anime_Name)
                    print(games)
                elif risposta_class == '3':
                    print("\nList of Rankings present:\n")
                    developers = ontology.search(is_a = ontology.Ranking)
                    print(developers)
                elif risposta_class == '4':
                    print("\nList of Genres present:\n")
                    genres = ontology.search(is_a = ontology.Genre)
                    print(genres)
                elif risposta_class == '5':
                    print("\nList of Types present:\n")
                    platforms = ontology.search(is_a = ontology.Type)
                    print(platforms)
                elif risposta_class == '6':
                    print("\nList of votes present:\n")
                    publishers = ontology.search(is_a = ontology.Vote)
                    print(publishers)
                else:
                    print("\nEnter the number correctly among those presented")

                print("\nWould you like to go back or continue?\n Back (yes) Continue (no)")
                risp = input("\n")
                if risp == 'yes':
                    break

        elif risp_menù == '2':
            print("Object properties present in the ontology:\n")
            print(list(ontology.object_properties()), "\n")
        elif risp_menù == '3':
            print("Properties of the data present in the ontology:\n")
            print(list(ontology.data_properties()), "\n")
        elif risp_menù == '4':
            print("Example queries:")
            print("List of anime featuring the 'Fantasy' category:\n")
            games = ontology.search(is_a = ontology.Anime_Name, has_genre = ontology.search(is_a = ontology.Fantasy))
            print(games, "\n")
            print("List of anime featuring the Ranking 'between 8-9':\n")
            games = ontology.search(is_a = ontology.Anime_Name, has_ranking = ontology.search(is_a = ontology.between_8_9))
            print(games, "\n")
            print("List of animex featuring the user 'user_38':\n")
            games = ontology.search(is_a = ontology.Anime_Name, has_user = ontology.search(is_a = ontology.user_38))
            print(games, "\n")
        elif risp_menù == '5':
            break