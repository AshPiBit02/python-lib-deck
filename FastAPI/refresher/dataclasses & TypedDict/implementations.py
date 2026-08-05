from dataclasses import dataclass

# Basic dataclass
@dataclass
class Book:
    title:str
    author:str
    pages:int
    available:bool=True
b1=Book("Into The Wild","James Nikol",579)
b2=Book("Atomic Habbits","James Clear",693,False)
print(b1)
print(b2)

# Equality check
b3=Book("Into The Wild","James Nikol",579)
print(b1==b3)

# Mutable default gotcha
from dataclasses import field
@dataclass 
class Cart:
    items:list[str]=field(default_factory=list)
cart1=Cart()
cart2=Cart()
cart1.items.append("This is the first cart")
print(cart1)
print(cart2)

# Frozen dataclass
@dataclass(frozen=True)
class Credentials:
    account_no:int
    balance:float
cre1=Credentials(444,983.22)
# cre1.account_no=434 # error(immutable) 

from typing import List, TypedDict
# Basic TypedDict
class Movie(TypedDict):
    title:str
    year:int
    rating:float
movie:dict={"title":"Spider-Man: Brand New Day","year":2026,"rating":9.8}
print(movie["title"])
print(movie)
fakemovie:dict={"title":"Spider-Man: Brand New Day","year":2026,"rating":"full"}
print(fakemovie["rating"]) # is invalid but doesn't raise error because python ignores type hints at runtime

# Optional Keys in TypedDict
from typing import NotRequired
class UserProfile(TypedDict):
    username:str
    email:str
    bio:NotRequired[str]
u1:UserProfile={"username":"Lianna","email":"lianna@gmail.com"}
u2:UserProfile={"username":"Hodor","email":"holdthedoor@gmail.com","bio":"Coder & learner"}
print(u1)
print(u2)

# dataclass to dict conversion and viceversa
from dataclasses import asdict
book1Dict:dict=asdict(b1)
print(book1Dict["title"])

# back to dataclass
b1New=Book(**book1Dict)
print(b1New)


# Combined dataclass+TypedDict+async
import asyncio

# Raw incoming data schema
class RawStudentData(TypedDict):
    name:str
    socre:int

# processed internal object
@dataclass
class Student:
    name:str
    score:int
    passed:bool

async def process_student(raw:RawStudentData)->Student:
    print("Processing raw student data.....")
    await asyncio.sleep(1)
    passed=raw["score"]>=60
    return Student(name=raw["name"],score=raw["score"],passed=passed)

async def process_all(raw_students:List[RawStudentData])->List[Student]:
    results=await asyncio.gather(*(process_student(r) for r in raw_students))
    return results

async def main()->None:
    raw_students=[
         {"name": "Alice", "score": 85},
        {"name": "Bob", "score": 40},
        {"name": "Charlie", "score": 72},
        {"name": "Diana", "score": 55},
    ]
    students=await process_all(raw_students)
    for s in students:
        print(s)

asyncio.run(main())