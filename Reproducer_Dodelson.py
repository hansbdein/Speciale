#python3 -i /home/hansbdein/Speciale/Hyggehejsa.py
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(1, 10, 100)
y = np.exp(x)

plt.plot(x, y)
plt.yscale('log')


plt.xlabel('x')
plt.ylabel('y')



plt.show()
plt.savefig("test.pdf")
print("Hello World")