from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'jyotirmay'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing student cgpa')

new_student = {'age':23, 'email':'abc@gmail.com'}

student = Student(**new_student)

print(type(student))
print(student)


student_dict = dict(student)
print(student_dict['age'])

student_json = student.model_dump_json()
print(student_json)

