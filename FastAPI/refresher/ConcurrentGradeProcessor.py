from dataclasses import dataclass
from typing import List
import asyncio

@dataclass
class Student:
    name:str
    grade:int
    passed:bool

students:List[Student]=[
    Student("Alice", 85, True),
    Student("Bob", 40, False),
    Student("Charlie", 72, True),
    Student("Diana", 55, False),
]

# Filtering 
passed_students=[s for s in students if s.passed]
failed_students=[s for s in students if not s.passed]

# Summary
print("-"*5," Summary ","-"*5)
for s in students:
    status="PASS" if s.passed else "Fail"
    print(f"{s.name}: Grade:{s.grade}, Status={status}")

# Async simulation
async def fetch_data(student:Student)->str:
    await asyncio.sleep(1)
    return f"Fetched record for {student.name}"

async def main():
    tasks=[fetch_data(s) for s in students]
    results=await asyncio.gather(*tasks)
    for r in results:
        print(r)

asyncio.run(main())