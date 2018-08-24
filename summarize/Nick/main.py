from reduction import *

reduction = Reduction()
text = open('may.txt').read()
reduction_ratio = 0.1
reduced_text = reduction.reduce(text, 6)
print(reduced_text)
