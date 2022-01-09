import zipfile as zf
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import requests


def extract_all(i_path, o_path):
    with zf.ZipFile(i_path, "r") as zip_f:
        zip_f.extractall(o_path)


def get_dataset(artists, albums, tracks):
    df_artists = artists.merge(albums, how="inner", on="artist_id", suffixes=('_artists', '_albums'))
    dataset = df_artists.merge(tracks, how="inner", on=["artist_id", "album_id"],
                               suffixes=('_artists_albums', '_tracks'))
    dataset = dataset.rename(columns={"name": "name_tracks", "popularity": "popularity_tracks"})
    dataset["name_artists"] = dataset["name_artists"].str.title()
    popularity_count = dataset["popularity_tracks"].isnull().values.sum()
    mean_popularity_tracks = dataset[dataset["popularity_tracks"].notnull()]["popularity_tracks"].values.mean()
    dataset = dataset.fillna(value={"popularity_tracks": mean_popularity_tracks})
    return popularity_count, dataset


def get_column_pandas(ruta, column):
    time_execution = time()
    df = pd.read_csv(ruta, sep=";")
    return list(df[column]), time() - time_execution


def get_column_mine(ruta, feature):
    time_execution = time()
    file = open(ruta, "r", encoding="utf8")
    lines = [line for line in file]
    file.close()
    column = list(lines[0].split(";")).index(feature)
    return [lines[x].split(";")[column] for x in range(1, len(lines))], time() - time_execution


def test_both(ruta, column):
    x, timePandas = get_column_pandas(ruta, column)
    y, timeMine = get_column_mine(ruta, column)
    assert (len(x) == len(y))
    analyzed = len(y)
    plt.ylabel('Execution time in seconds')
    plt.title(f'Time of execution of {column} with {analyzed} elementos')
    plt.xlabel("Rows")
    plt.plot(len(x), timePandas, "p", label="Pandas")
    plt.plot(len(x), timeMine, "p", label="Mine")
    plt.legend()
    plt.show()


def tracks_by_filter(df, column, f):
    return df[column].astype(str).str.contains(f).sum()


def popularity_track_filter_year(df, r):
    result = df.loc[df["release_year"].astype(int) >= r]
    return result.sort_values(by=["popularity_tracks"], ascending=False).head(1)


def artist_decade(df, year):
    result = df.loc[df["release_year"].astype(int) >= year]
    result["release_year"] = pd.to_datetime(result["release_year"], format='%Y').dt.year.floordiv(10).mul(10)
    return result[["name_artists", "name_tracks", "release_year"]]


def my_min(df, col):
    return df[col].min()


def my_max(df, col):
    return df[col].max()


def my_mean(df, col):
    return df[col].mean()


def mean_danceability(df, a):
    result = df.groupby(by=["name_artists", "name_albums"]).agg({'danceability': 'mean'}).reset_index()
    result = result.loc[result["name_artists"] == a]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots()
    result.plot(x="name_albums", y="danceability", ax=ax)
    ax.tick_params(axis='x', labelsize=6, labelrotation=15)
    plt.ylabel("danceability")
    plt.xlabel("name albums")
    plt.show()


def density_audio_feature(df, a, f):
    df = df[df["name_artists"].isin(a)]
    result = df[["artist_id", "name_artists", f]].groupby(by=["name_artists", f]).count().reset_index(). \
        rename(columns={"artist_id": "counter"})
    result["density"] = result["counter"] / result.shape[0]
    return result


def reformat_artist_name(artist):
    artist = artist.replace(' & ', " and ")
    artist = artist.replace('. ', ".")
    artist = artist.replace(' ', "%20")
    return artist

def get_data(artist):
    artist = reformat_artist_name(artist)
    response = requests.get(f"https://theaudiodb.com/api/v1/json/2/search.php?s={artist}")
    json = response.json()
    if json["artists"] is None:
        return "NAN", "NAN"
    return json["artists"][0]["intFormedYear"], json["artists"][0]["strCountry"]


def fill_audiodb_csv():
    time_execution = time()
    df = pd.read_csv("data/artists_norm.csv", sep=";")
    columns = ["artist_name", "formed_year", "country"]
    elementos = []
    for x in range(len(df["name"])):
        artist_name = df["name"][x]
        formed_year, country = get_data(artist_name)
        elementos.append([artist_name, formed_year, country])
    endDf = pd.DataFrame(elementos, columns=columns)
    endDf.to_csv("data/artist_audiodb.csv")
    return time() - time_execution