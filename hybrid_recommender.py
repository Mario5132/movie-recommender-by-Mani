import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load data
movies = pd.read_csv('Data/movies_metadata.csv', low_memory=False)
# Rename column 'id' to 'movieId' so it matches ratings.csv
movies.rename(columns={'id': 'movieId'}, inplace=True)
ratings = pd.read_csv('Data/ratings_small.csv')

from ast import literal_eval

# Extract main actor and director
main_actor = []
main_director = []

for i, row in movies.iterrows():
    # Main actor
    try:
        main_actor.append(literal_eval(row['cast'])[0]['name'])
    except:
        main_actor.append(None)
    # Director
    try:
        directors = [x['name'] for x in literal_eval(row['crew']) if x['job'] == 'Director']
        main_director.append(directors[0] if directors else None)
    except:
        main_director.append(None)

movies['main_actor'] = main_actor
movies['main_director'] = main_director

# Combine genres + main actor + main director into one string
movies['combined_features'] = movies['genres'].fillna('') + ' ' + \
                              movies['main_actor'].fillna('') + ' ' + \
                              movies['main_director'].fillna('')



movies = movies.head(5000)

movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')

# Drop rows with invalid movieId
movies = movies.dropna(subset=['movieId'])
ratings = ratings.dropna(subset=['movieId'])

# Convert to int
movies['movieId'] = movies['movieId'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)


# Merge data
data = pd.merge(ratings, movies, on='movieId')

# Create content similarity based on combined features (genres + actor + director)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies['combined_features'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)


def get_recommendations(title, cosine_sim=cosine_sim):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx = indices.get(title)

    if idx is None:
        return f"Movie '{title}' not found."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Example run
if __name__ == "__main__":
   print("Recommended movies for 'Toy Story':")
   print(get_recommendations('Toy Story'))
