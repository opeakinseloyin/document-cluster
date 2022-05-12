import pandas as pd
from joblib import dump
from preprocess import preprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


df = pd.read_csv("Book1.csv", encoding="utf-8")

df['new'] = df['Document'].apply(preprocessor)


document = df['new'].values.astype('U')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(document)



model = KMeans(n_clusters=3, random_state=101)
model.fit(X)

df['clusters'] = model.labels_

print(model.labels_)

print("Cluster centroids: \n")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(3):
    print("Cluster %d:" % i)
    for j in order_centroids[i, :10]: #print out 10 feature terms of each cluster
        print (' %s' % terms[j])
    print('------------')


dump(model, 'cluster_model.joblib')
dump(vectorizer, 'vectorizer.joblib')