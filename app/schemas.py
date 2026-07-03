from pydantic import BaseModel, Field

class CreditRequest(BaseModel):
    person_age: int = Field(..., example=22)
    person_income: float = Field(..., example=59000.0)
    person_home_ownership: str = Field(..., example="RENT")
    person_emp_length: float = Field(..., example=4.0)
    loan_intent: str = Field(..., example="PERSONAL")
    loan_grade: str = Field(..., example="D")
    loan_amnt: float = Field(..., example=35000.0)
    loan_int_rate: float = Field(..., example=16.02)
    loan_percent_income: float = Field(..., example=0.59)
    cb_person_default_on_file: str = Field(..., example="Y")
    cb_person_cred_hist_length: int = Field(..., example=3)