from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "ash"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description="A decimal value giving the cgpa of student")

new_student = {
    'name': 'ashish',
    'age': 32,
    'email': 'ash@gmail.com',
    'cgpa': 1
}

student = Student(**new_student)

print(student)
print(student.name)
print(type(student))
