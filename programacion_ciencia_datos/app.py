# Importamos librerías
from datetime import datetime
import requests
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from transf_functions import extract_all, get_dataset, tracks_by_filter, popularity_track_filter_year, \
    artist_decade, my_min, my_max, my_mean, mean_danceability, density_audio_feature, test_both, fill_audiodb_csv

# Guardamos las features
features = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"]

# Ejercicio 1

def exercise_i():
    """
    Extracción y limpieza de los datos
    :param ""
    :return: nº tracks total y columnas, nº tracks no tiene valor de popularity

    """
    extract_all("data.zip", "data/")
    albums = pd.read_csv("data/albums_norm.csv", sep=";")
    artists = pd.read_csv("data/artists_norm.csv", sep=";")
    tracks = pd.read_csv("data/tracks_norm.csv", sep=";")
    popularity_count, dataset = get_dataset(artists, albums, tracks)
    print(f"Tracks and Columns: {dataset.shape}")
    print(f"Tracks Without Popularity: {popularity_count}")
    return dataset


# Ejercicio 2
def exercise_ii():
    """
    Implemtación de lectura de datos con pandas y una propia
    :param ""
    :return: gráfico comparativo de las dos versiones
    """
    test_both("data/artists_norm.csv", "artist_id")
    test_both("data/albums_norm.csv", "album_id")
    test_both("data/tracks_norm.csv", "track_id")

# Ejercicio 3
def exercise_iii(df):
    """
    Exploración de datos
    :param df:
    :return: responde a las preguntas elaboradas para el ejercicio 3
    """
    result = tracks_by_filter(df, "name_artists", "Radiohead")
    print(f"Tracks of Radiohead {result}")

    result = tracks_by_filter(df, "name_tracks", "police")
    print(f"Tracks with police {result}")

    result = tracks_by_filter(df, "release_year", "^199")
    print(f"Tracks from albums in the 1990s {result}")

    year = datetime.now().year - 10
    result = popularity_track_filter_year(df, year)
    print(f"Track most popular of the last 10 years {result.name_tracks.values[0]}")

    result = artist_decade(df, 1960)
    print(f"\n {result['name_artists'].unique()}")

# Ejercicio 4
def exercise_iv(df):
    """
    Análisis estadístico de datos
    :param df:
    :return: estadistica básicas del apartado A y gráfico del apartado B
    """
    result = df[df["name_artists"] == "Metallica"]
    print(f"Min: {my_min(result, 'energy')} Max: {my_max(result, 'energy')} Mean: {my_mean(result, 'energy')}")
    mean_danceability(df, "Coldplay")

# Ejercicio 5
def exercise_v(df, a, f):
    """
    Plot del anterior apartado
    :param df: dataframe
    :param a:artists
    :param f: features
    :return: histograma de lo solicitado
    """
    result = density_audio_feature(df, a, f)
    result.hist(column=f, alpha=0.7, color="royalblue")
    plt.ylabel("density")
    plt.title("exercise_v")
    plt.grid(False)
    plt.show()

# Ejercicio 6
def exercise_vi(df, a, f):
    """
    Comparar artistas visualmente
    :param df: dataframe
    :param a:artists
    :param f: features
    :return: histograma de energy de Adele y Extremoduro
    """

    result = density_audio_feature(df, a, f)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    adele = result[result["name_artists"] == a[0]].rename(columns={f: f"{f}_{a[0]}"})
    adele.hist(column=f"{f}_{a[0]}", ax=ax, label="Adele")
    extremoduro = result[result["name_artists"] == a[1]].rename(columns={"energy": f"{f}_{a[1]}"})
    extremoduro.hist(column=f"{f}_{a[1]}", ax=ax, alpha=0.5, label="Extremoduro")
    plt.ylabel("density")
    plt.legend(loc='upper center')
    plt.title("exercise_vi")
    plt.grid(False)
    plt.show()

# Ejercicio 7
def exercise_vii(df, artists):
    """
    Calcular similitud entre artistas
    :param df:
    :param artists:
    :return: mapa de calor utilizando similitud euclidiana y cosinus
    """
    df_mean_feature = df.groupby(by=["name_artists"]).agg({
        "danceability": "mean",
        "energy": "mean",
        "key": "mean",
        "loudness": "mean",
        "mode": "mean",
        "speechiness": "mean",
        "acousticness": "mean",
        "instrumentalness": "mean",
        "liveness": "mean",
        "valence": "mean",
        "tempo": "mean"
    }).reset_index()
    df_mean_feature = df_mean_feature.loc[df_mean_feature["name_artists"].isin(artists)]
    # cosine_similarity
    result = cosine_similarity(df_mean_feature[features])
    result = pd.DataFrame(result, columns=df_mean_feature["name_artists"], index=df_mean_feature["name_artists"])
    sns.heatmap(result, annot=True, fmt=".3g")
    plt.title("Cosine similarity")
    plt.show()

    # Euclidean distance score and similarity
    result = euclidean_distances(df_mean_feature[features])
    result = pd.DataFrame(result, columns=df_mean_feature["name_artists"], index=df_mean_feature["name_artists"])
    sns.heatmap(result, annot=True, fmt=".3g")
    plt.title("Euclidean distance")
    plt.show()

# Ejercicio 8
def exercise_viii():
    """
    Llamadas a API externa
    :param ""
    :return: mostrar por pantalla los resultados obtenidos
    """

    time_execution = fill_audiodb_csv()
    df = pd.read_csv("data/artist_audiodb.csv")
    artist = ["Radiohead", "David Bowie", "Måneskin"]
    for x in range(len(df["artist_name"])):
        if df['artist_name'][x] in artist:
            print(f"Artist {df['artist_name'][x]} start at year {df['formed_year'][x]} and born in {df['country'][x]}")
    print(f"Total time: {round(time_execution, 2)} seconds")


if __name__ == '__main__':
    df = exercise_i()
    exercise_ii()
    exercise_iii(df)
    exercise_iv(df)
    exercise_v(df, ["Ed Sheeran"], "acousticness")
    exercise_vi(df, ["Adele", "Extremoduro"], "energy")
    exercise_vii(df, ["Metallica", "Extremoduro", "Ac/Dc", "Hans Zimmer"])
    exercise_viii()
