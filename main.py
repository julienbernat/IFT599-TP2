import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_partial_matrix(matrix, rows, cols, title):
    print('visualize_partial_matrix')
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix.iloc[:rows, :cols], cmap='viridis', linewidths=.5)
    plt.title(title)
    plt.show()

def plot_genre_distribution(movies):
    print('plot_genre_distribution')
    # Faire une copie explicite pour éviter le SettingWithCopyWarning
    movies_copy = movies.copy()
    movies_copy = movies_copy[movies_copy['genres'] != '(no genres listed)']

    genres_count = movies_copy['genres'].str.split('|', expand=True).stack().value_counts()

    # Créer le diagramme en bâton
    plt.figure(figsize=(12, 6))
    genres_count.plot(kind='bar', color='skyblue')
    plt.title('Nombre de films par genre (excluant "(no genres listed)")')
    plt.xlabel('Genre')
    plt.ylabel('Nombre de films')
    plt.xticks(rotation=45, ha='right')
    plt.show()

def create_new_datasets(movies, ratings):
    print('create_new_datasets')
    # Faire une copie explicite pour éviter le SettingWithCopyWarning
    movies_copy = movies.copy()

    # Exclure les films au genre non listé
    movies_copy = movies_copy[movies_copy['genres'] != '(no genres listed)']

    # Faire une copie explicite pour éviter le SettingWithCopyWarning
    ratings_copy = ratings.copy()

    # Filtrer les évaluations en fonction des films retenus
    ratings_copy = ratings_copy[ratings_copy['movieId'].isin(movies_copy['movieId'])]

    # Ajuster les valeurs de rating selon les spécifications
    ratings_copy['rating'] = ratings_copy['rating'].map({5.5: 5, 5.0: 5, 4.5: 4, 4.0: 4, 3.5: 3, 3.0: 3, 2.5: 2, 2.0: 2, 1.5: 1, 1.0: 1, 0.5: 1, 0.0: 0})

    # Enregistrer les nouveaux jeux de données
    movies_copy.to_csv('movies1.csv', index=False)
    ratings_copy.to_csv('ratings1.csv', index=False)

def create_content_matrix(movies):
    print('create_content_matrix')
    # Créer une liste unique de genres
    unique_genres = sorted(list(set(genre for genres in movies['genres'].str.split('|') for genre in genres)))

    # Initialiser la matrice binaire avec des zéros
    content_matrix = pd.DataFrame(0, index=movies['movieId'], columns=unique_genres)

    # Remplir la matrice avec des uns pour les genres associés à chaque film
    for index, row in movies.iterrows():
        if index % 1000 == 0:
            print(index)
        genres = row['genres'].split('|')
        content_matrix.loc[row['movieId'], genres] = 1

    # Enregistrer la matrice de contenu dans un fichier CSV
    content_matrix.to_csv('content_matrix.csv')

    # Retourner la matrice de contenu
    return content_matrix

def create_user_profile_matrix(ratings, content_matrix):
    print('create_user_profile_matrix')

    # Merge ratings and content_matrix on movieId to get genres for each rating
    merged_df = pd.merge(ratings, content_matrix, left_on='movieId', right_index=True)
    merged_df = merged_df.sort_values(by='userId')
    merged_df.to_csv('merged_df.csv')

    # Initialize the user profile matrix with zeros
    user_profile_matrix = pd.DataFrame(0, index=ratings['userId'], columns=content_matrix.columns)

    # Iterate through each row (rating) in the merged dataframe
    for index, row in merged_df.iterrows():
        user_id = row['userId']
        rating = row['rating']
        content_vector = row.iloc[3:]  # Assuming the content features start from the 4th column
        if(user_id % 1000 == 0):
            print('user_id', user_id)
        # Update the user profile matrix using the linear combination formula
        user_profile_matrix.loc[user_id] += rating * content_vector

    # Save the user_profile_matrix to a CSV file
    user_profile_matrix.to_csv('user_profile_matrix.csv')

    print('Updated user_profile_matrix saved to user_profile_matrix_updated.csv')
    return user_profile_matrix


def main():
    movies = pd.read_csv('movies.csv')    
    ratings = pd.read_csv('ratings.csv')
    
    # Partie 1: Créer le diagramme en bâton pour le nombre de films par genre
    # plot_genre_distribution(movies)

    # Partie 2: Créer les nouveaux jeux de données movies1.csv et ratings1.csv
    # create_new_datasets(movies, ratings)

    movies1 = pd.read_csv('movies1.csv')    
    ratings1 = pd.read_csv('ratings1.csv')

    # Partie 3: Créer la matrice binaire de contenu C caractérisant chaque film
    # create_content_matrix(movies1)

    content_matrix = pd.read_csv('content_matrix.csv', index_col=0)
    # visualize_partial_matrix(content_matrix, 10, 19, 'Matrice binaire de contenu C')

    # Partie 4: Construire la matrice de profil des utilisateurs P
    user_profile_matrix = create_user_profile_matrix(ratings1, content_matrix)
    user_profile_matrix = pd.read_csv('user_profile_matrix.csv', index_col=0)
    visualize_partial_matrix(user_profile_matrix, 10, 19, 'Matrice de profil des utilisateurs P')

if __name__ == "__main__":
    main()
