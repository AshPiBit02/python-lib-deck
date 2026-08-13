from pydantic import BaseModel,model_validator

class Book(BaseModel):
    copies_available:int
    borrowed_by:list[str]=[]

    @model_validator(mode="after")
    def check_consistency(self)->"Book":
        if self.copies_available == 0 and not self.borrowed_by:
            raise ValueError("copies_available is 0 but no borrowers are recorded")
        return self

# Book(copies_available=0,borrowed_by=[])
Book(copies_available=0,borrowed_by=["Aegon","Eddard"])
Book(copies_available=3,borrowed_by=[])