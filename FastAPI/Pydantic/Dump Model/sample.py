from pydantic import BaseModel

class Book(BaseModel):
    title:str
    author:str=None
    copies_available:int|None=None

book=Book(title="Dune",author="Frank Herbert")
print(book.model_dump())
print(book.model_dump_json())

print(book.model_dump(exclude={"copies_available","author"})) # drop specific fields
print(book.model_dump(include={"title","author"}))
print(book.model_dump(exclude_unset=True)) # only fields the CLIENT actually provided
print(book.model_dump(exclude_none=True))
# print(Book.model_dump(exclude_defaults=True)) exclude fields whose value equals the field's defaults
print(book.model_dump(exclude_none=True)) # excludes fields whose value is None