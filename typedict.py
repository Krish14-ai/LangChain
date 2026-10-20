from typing import TypedDict

class Person(TypedDict):
    name : str
    age : int

new_perosn : Person = {'name' : 'Krish', 'age' : 20}

print(new_perosn)