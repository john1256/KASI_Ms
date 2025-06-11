import os
import numpy as np
CurrentPath = os.getcwd()
print(CurrentPath)
dir = os.path.dirname(__file__)
print("Test completed successfully.")
with open('./Results/hello.txt', 'w') as f:
    f.write('Hello, world!')
