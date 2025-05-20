import psycopg2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from db import get_connection

def load_training_data():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT h, s, v, label FROM batik_features")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Pisahkan fitur dan label
    features = [row[:3] for row in rows]
    labels = [row[3] for row in rows]
    return np.array(features), np.array(labels)

def classify_hsv_feature(hsv_feature, k=3):
    features, labels = load_training_data()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)
    prediction = knn.predict([hsv_feature])
    return prediction[0]
