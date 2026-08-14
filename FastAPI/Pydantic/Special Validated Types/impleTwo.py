from pydantic import BaseModel,SecretStr
class RecruiterAccount(BaseModel):
    username:str
    api_key:SecretStr

acc1=RecruiterAccount(username="Hedge knight",api_key="KGHT-HGE-89")
print(acc1)
print(acc1.model_dump())