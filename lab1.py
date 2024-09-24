import math
import random



class Neurone:
    def __init__(self, n):
        self.number = n
        self.weight = []
        self.somma = 0
    def set_w(self, w = []):
        if len(w) == 0:
            self.weight = [random.randint(-10,10) for _ in range(self.number)]
        else:
            self.weight = w
    def prevedere(self, x, func):
        self.somma = 0
        if len(x) != len(self.weight):
            return "Number weight =! number inputs"
        for i in range(self.number):
            self.somma += x[i] * self.weight[i]
        return func(self.somma)


def impl(a):
    return 0 if a > 0 else 1

def cos(a):
    return math.cos(a)

x = int(input("Enter number of inputs -> "))
dd = Neurone(x)

def relu(a):
    if a > 0 :
        return a
    else:
        return 0

    
X = [[] for i in range(100)]
I = 0

for i in X:
    I+=1
    i.append(math.sin(math.pi * 2 + math.pi/4 + I*I) + math.pi )
    i.append(I)
    i.append(I * math.sin(math.pi*2+math.pi/2 + I))

dd.set_w([1, 1, 1])

for i in range(100):
    print(X[i],"  ",dd.prevedere(X[i],relu))

for _ in range(10):
    inp = [random.randint(0,1) for _ in range(x)]
    print(inp,"",len(inp),dd.prevedere(inp, impl))


