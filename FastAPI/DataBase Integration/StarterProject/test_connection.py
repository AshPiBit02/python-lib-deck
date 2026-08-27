from db.database import engine

with engine.connect() as conn:
    print("Connected to ",conn.engine.url)