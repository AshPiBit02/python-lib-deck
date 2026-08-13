from pydantic import model_validator,BaseModel
from datetime import date
class DataRange(BaseModel):
    start_date:date
    end_date:date

    @model_validator(mode="after")
    def check_date(self)->"DataRange":
        if self.start_date>self.end_date:
            raise ValueError("end_date never be before the start_date!")
        return self

DataRange(start_date=date(2026,2,13),end_date=date(2026,2,13))
# DataRange(start_date=date(2026,2,13),end_date=date(2026,1,13)) # error