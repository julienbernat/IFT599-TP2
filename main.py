import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

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

    # Group by 'userId' and aggregate the values
    grouped_df = merged_df.groupby('userId').agg(lambda x: sum(x) if x.name in content_matrix.columns else x.iloc[0])

    # Extract relevant columns for the user profile matrix
    relevant_columns = content_matrix.columns.intersection(grouped_df.columns)
    
    # Create the user profile matrix directly
    user_profile_matrix = pd.DataFrame(grouped_df['rating'].values[:, None] * grouped_df[relevant_columns].values,
                                       index=grouped_df.index,
                                       columns=relevant_columns)

    # Save the user_profile_matrix to a CSV file
    user_profile_matrix.to_csv('user_profile_matrix.csv')

    return user_profile_matrix



def spectral_clustering(user_profile_matrix):

    # Convert user_profile_matrix to 2D for visualization (you may need to adjust columns)
    user_profile_2d = user_profile_matrix.iloc[:, :2]

    # Perform spectral clustering with different values of k
    k_values = [2, 3, 4, 5]

    scores = []
    for k in k_values:
        # Apply spectral clustering
        clustering = SpectralClustering(n_clusters=k, random_state=42)
        labels = clustering.fit_predict(user_profile_2d)

        # Compute silhouette score
        silhouette_avg = silhouette_score(user_profile_2d, labels)
        scores.append(silhouette_avg)

        # Visualize the clusters
        plt.scatter(user_profile_2d.iloc[:, 0], user_profile_2d.iloc[:, 1], c=labels, cmap='viridis')
        plt.title(f'Spectral Clustering (k={k}), Silhouette Score: {silhouette_avg:.2f}')
        plt.show()

    # Plot silhouette scores
    plt.plot(k_values, scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Spectral Clustering')
    plt.show()


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
    visualize_partial_matrix(content_matrix, 30, 19, 'Matrice binaire de contenu C')

    # Partie 4: Construire la matrice de profil des utilisateurs P
    # user_profile_matrix = create_user_profile_matrix(ratings1, content_matrix)

    user_profile_matrix = pd.read_csv('user_profile_matrix.csv', index_col=0)
    visualize_partial_matrix(user_profile_matrix, 30, 19, 'Matrice de profil des utilisateurs P')

    # Partie 5: Clustering spectral
    spectral_clustering(user_profile_matrix)


if __name__ == "__main__":
    main()
