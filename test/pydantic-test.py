from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

person = Person(name="John", age=30)
print(person.model_dump_json())