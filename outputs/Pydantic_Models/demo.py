from pydantic import BaseModel

class Students(BaseModel) :
    
    name : str 
    
krish = {'name' : 'krish'}
new_student = Students(**krish)

print(new_student)