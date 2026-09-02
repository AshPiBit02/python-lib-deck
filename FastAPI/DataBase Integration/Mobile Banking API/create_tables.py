from db.database import Base,engine
from models import (
    Customer,Account,
    AccountCustomer,Card,AuditLog,Transaction)
print("Creating tables...")
Base.metadata.create_all(bind=engine)