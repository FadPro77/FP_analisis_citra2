# db.py
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS")
    )

def insert_feature_to_db(image_name, label, hsv_features):
    conn = get_connection()
    print('connected')

    cur = conn.cursor()
    cur.execute("""
        INSERT INTO batik_features (image_name, label, h, s, v)
        VALUES (%s, %s, %s, %s, %s)
    """, (image_name, label, *hsv_features))
    conn.commit()
    cur.close()
    conn.close()
