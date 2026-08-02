def greet(name:str,age:int)->str:
    if age<18:
        return f"Hello {name}, you are naive."
    else:
        return f"Hello {name}, you are inexperienced."
print(greet("Aashish",21)) 

def add(num1:int,num2:int)->int:
    return num1+num2

# Default parameters
def greet(name:str="Sir",greeting:str="Hello")->str:
    return f"{greeting}!, {name}"

# print(greet())
# print(greet("Aashish"))

# *args usage
def multiply_all(*args:int)->int:
    product=1
    for num in args:
        product*=num
    return product

print(multiply_all(2,3,4))


# Any

from typing import Any
def add_all(*args:Any)->Any:
    result="fasd"
    for values in args:
        result+=values
    return result

print(add_all("sd","asf","fasd"))

def demo(*args:Any)->None:
    for item in args:
        print(f"Value:{item},Type:{type(item).__name__}")

demo(10,"Aashish",[1,2,3],True)


# *kwargs usage
from typing import Union,Dict
def student_info(**kwargs)->Dict[str,Union[str,int]]:
    return kwargs
print(student_info(name=["Jon","Rob"],age=[21,23]))


# Task1
def grade_students(name:str,marks:list[int])->Dict[str,object]:
    total_marks=sum(marks)
    avg=total_marks/len(marks)
    passed=avg>40
    return{
        "Name":name,"Average":avg,"Passed":passed
    }

print(grade_students("Jon",[90,89,94]))
print(grade_students("Rob",[50,29,34]))

        
