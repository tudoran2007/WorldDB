from nn import *
from matplotlib import pyplot as plt

generations = 2000
goobercount = 50
generationlength = 1000
template = None
fitnessovergenerations = []
bests = []
bestfitness = 0

#i = 0
for i in range(generations):
#while True:
    #i+=1
    print(i+1)
    generation = Generation(goobercount, generationlength, template)
    generation.runsim()
    template, fitness, fittestingen = generation.returnfittest()
    #print(i+1, fitness)
    fitnessovergenerations.append(fitness)
    bests.append(fittestingen)

ageovergenerations = []
for i in range(len(bests)):
    ageovergenerations.append(bests[i].state["age"])

print(bests[-1].state)
plt.plot(fitnessovergenerations)
plt.plot(ageovergenerations)
plt.show()