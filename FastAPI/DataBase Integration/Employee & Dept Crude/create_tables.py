from db.database import Base,engine
from models.department import Department
from models.employee import Employee

Base.metadata.create_all(bind=engine)