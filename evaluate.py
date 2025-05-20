import psycopg2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_training_data():
    conn = psycopg2.connect(
        dbname="batikdb",
        user="postgres",
        password="password",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute("SELECT h, s, v, label FROM batik_features")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    features = [row[:3] for row in rows]
    labels = [row[3] for row in rows]
    return np.array(features), np.array(labels)

def evaluate_knn(k=3):
    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, xticklabels=np.unique(y), yticklabels=np.unique(y), cmap="Blues", fmt='g')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate_knn()
