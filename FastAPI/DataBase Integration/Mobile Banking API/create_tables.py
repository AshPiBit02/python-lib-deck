from db.database import Base,engine
from models import Customer
print("Creating tables...")
Base.metadata.create_all(bind=engine)