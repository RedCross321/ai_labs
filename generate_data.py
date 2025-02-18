import numpy as np

# Generate X values from -20 to 20 with step 0.1
x = np.arange(-20, 20.1, 0.1)
d = x.reshape(-1, 1)
y = np.sin(x) + np.sin(np.sqrt(2) * x)

N = np.floor(x / (2 * np.pi))

b = x % (2 * np.pi)

np.savetxt('data.csv', np.column_stack((d, N, b, y)), delimiter=',', header='X,N,b,Y', comments='', fmt='%.6f')
