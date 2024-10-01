import math
import random
class Neurone:
    def __init__(self, n):
        self.number = n
        self.weight = []
        self.somma = 0
    def set_w(self, w = []):
        if len(w) == 0:
            self.weight = [random.randint(-5,5) for _ in range(self.number)]
        else:
            self.weight = w
    def prevedere(self, x, func):
        self.somma = 0
        if len(x) != len(self.weight):
            return "Number weight =! number inputs"
        for i in range(self.number):
            self.somma += x[i] * self.weight[i]
        return func(self.somma)


class NeuralNetwork(Neurone):
    def __init__(self, n, x):
        self.number = n
        self.input = x
        self.dd = []
        for i in range(self.number):
            self.dd.append(f'a{i}')
    def predict(self, inp):
        #print(self.dd)
        #inp = [random.randint(1,5) for _ in range(self.input)]
        
        for i in range(self.number):
            self.dd[i] = Neurone(self.input)
            self.dd[i].set_w()
            print(inp,"", self.dd[i].prevedere(inp, chet), self.dd[i].weight)

def impl(a):
    return 0 if a > 0 else 1

def cos(a):
    return math.cos(a)

def chet(a):
    return a

inp = [1, 3]

Neur_n = int(input("Enter number of neurone -> "))
num_input = int(input("Enter number of input -> "))
run = NeuralNetwork(Neur_n, num_input)

run.predict(inp)















# def relu(a):
#     if a > 0 :
#         return a
#     else:
#         return 0

    
# X = [[] for i in range(100)]
# I = 0

# for i in X:
#     I+=1
#     i.append(math.sin(math.pi * 2 + math.pi/4 + I*I) + math.pi )
#     i.append(I)
#     i.append(I * math.sin(math.pi*2+math.pi/2 + I))

# dd.set_w([1, 1, 1])

# for i in range(100):
#     print(X[i],"  ",dd.prevedere(X[i],relu))

# for _ in range(10):
#     inp = [random.randint(0,1) for _ in range(x)]
#     print(inp,"",len(inp),dd.prevedere(inp, impl))


