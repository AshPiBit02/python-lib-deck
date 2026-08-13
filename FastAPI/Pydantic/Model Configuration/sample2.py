from pydantic import BaseModel,ConfigDict

class AppBaseModel(BaseModel):
    model_config=ConfigDict(
        str_strip_whitespace=True,
        extra="ignore",
        str_to_lower=True
    )

class BookIn(AppBaseModel):
    title:str
    author:str

class UserIn(AppBaseModel):
    username:str
    email:str

book=BookIn(title="   Atomic Habbits   ",author="James Clear ",price=156)
user=UserIn(username="ashpibit02  ",email="   aashishchaudhari240@gmail.com")
print(book)
print(user)