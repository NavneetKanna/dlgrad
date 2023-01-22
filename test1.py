from test2 import a

a.bg = True 

del a

from test2 import a

print(a.bg)