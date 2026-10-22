from typing import Dict

# Using Typed Dict to specify keys and values which helps the Dict to
# follow a Specific structure 
class Person(Dict):
    name : str
    age : int
    gender : str
    
me = Person({
    'name' : 'krish',
    'age' : 21,
    'gender' : 'male'
})

print(me)