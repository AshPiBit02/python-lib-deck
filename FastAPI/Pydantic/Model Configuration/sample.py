from pydantic import BaseModel,ConfigDict

class BookIn(BaseModel):

    # model_config={
    #     "str_strip_whitespace":True,
    #     "extra":"forbid"
    # }
    model_config=ConfigDict(
        str_strip_whitespace=True,
        extra="forbid"
    )
    title:str
    author:str

book1=BookIn(title="Sanke   ",author="  asdf ",ichbin="urf")
print(book1)