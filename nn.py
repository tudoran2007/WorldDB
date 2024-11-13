import random
import math
import copy

class Neuron:
    def __init__(self, inputcount, weights=None, bias=None):
        self.inputcount = inputcount
        self.weights = weights if weights else [random.uniform(-1, 1) for _ in range(inputcount)]
        self.bias = bias if bias else random.uniform(-1, 1)
        self.output = 0

    def calculate(self, inputs):
        self.output = sum([(float(inputs[i]) / 100) * self.weights[i] for i in range(self.inputcount)]) + self.bias
        self.output = 1 / (1 + (math.e) ** (-self.output))
        return self.output

    def returninfo(self):
        return [self.weights, self.bias]

class Brain:
    def __init__(self, inputcount, outputcount, template):
        self.init = template
        self.inputcount = inputcount
        self.outputcount = outputcount
        self.neurons = []
        if template is not None:
            for i in range(outputcount):
                self.neurons.append(Neuron(inputcount, template[i][0], template[i][1]))
        else:
            for i in range(outputcount):
                self.neurons.append(Neuron(inputcount))
        self.template = [neuron.returninfo() for neuron in self.neurons]

    def calculate(self, inputs):
        self.outputs = []
        for i in range(self.outputcount):
            self.outputs.append(self.neurons[i].calculate(inputs))
        return self.outputs

class Goober:
    def __init__(self, mutate, template=None):
        self.template = copy.deepcopy(template)
        self.state = {
            "hunger": 0,
            "fatigue": 0,
            "money": 10,
            "happiness": 50,
            "health": 100,
            "age": 0
        }
        self.actionset = ["eat", "sleep", "work", "mindless consumerism"]
        self.mutated = mutate
        inputcount = len(self.state)
        outputcount = len(self.actionset)
        if mutate == 1 and self.template is not None:
            mutatedneuron = random.randint(0, len(self.template) - 1)
            mutation = random.randint(0, 1)  # weight or bias
            if mutation == 0:  # weight
                mutatedweight = random.randint(0, len(self.template[mutatedneuron][0]) - 1)
                self.template[mutatedneuron][0][mutatedweight] = self.template[mutatedneuron][0][mutatedweight] + random.uniform(-0.2, 0.2)
            elif mutation == 1:  # bias
                self.template[mutatedneuron][1] = self.template[mutatedneuron][1] + random.uniform(-0.2, 0.2)
            for neuron in range(len(self.template)):
                self.template[neuron][1] = min(2.0, max(-2.0, self.template[neuron][1])) #map bias from -2 to 2
                for weight in range(len(self.template[neuron][0])):
                    self.template[neuron][0][weight] = min(2.0, max(-2.0, self.template[neuron][0][weight])) #map weight from -2 to 2
        self.brain = Brain(inputcount, outputcount, self.template)
        self.template = self.brain.template

    def decide(self):
        inputs = list(self.state.values())
        outputs = self.brain.calculate(inputs)
        highest = 0
        for i in range(len(outputs)):
            if outputs[i] > outputs[highest]:
                highest = i
            elif outputs[i] == outputs[highest]:
                if random.randint(0, 1) == 1:
                    highest = i
        return self.actionset[highest]

    def act(self, action):
        if action == "eat" and self.state["money"] >= 5:
            self.state["hunger"] -= 50
            self.state["money"] -= 5
        elif action == "sleep":
            self.state["fatigue"] -= 50
        elif action == "work":
            self.state["money"] += 2
            self.state["hunger"] += 5
            self.state["fatigue"] += 5
            self.state["happiness"] -= 2
        elif action == "mindless consumerism" and self.state["money"] >= 5:
            self.state["money"] -= 5
            self.state["happiness"] += 20
        
        # Increment hunger, fatigue, and modify happiness and health based on hunger/fatigue
        self.state["hunger"] += 2
        self.state["fatigue"] += 2
        self.state["happiness"] -= ((self.state["hunger"] - 50) / 2) if self.state["hunger"] > 50 else 1 + ((self.state["fatigue"] - 50) / 2) if self.state["fatigue"] > 50 else 1
        self.state["health"] += 3 - (((self.state["hunger"] - 50) / 5) if self.state["hunger"] > 50 else 0) - (((self.state["fatigue"] - 50) / 5) if self.state["fatigue"] > 50 else 0) - (5 if self.state["happiness"] < 1 else 0)
        self.state["age"] += 1

        for i in self.state.keys():
            if i != "age":
                self.state[i] = min(100, max(0, self.state[i]))


    def returnfitness(self):
        age_score = self.state["age"]*2
        health_score = self.state["health"]
        happiness_score = self.state["happiness"]
        hunger_penalty = (max(0, 100 - self.state["hunger"]) / 2) if self.state["hunger"] > 50 else 0
        fatigue_penalty = (max(0, 100 - self.state["fatigue"]) / 2) if self.state["fatigue"] > 50 else 0
        money_score = self.state["money"] / 10
        
        fitness = age_score + health_score + happiness_score + money_score - hunger_penalty - fatigue_penalty
        
        return fitness

class Generation:
    def __init__(self, goobercount, length, template=None):
        self.goobercount = goobercount
        self.length = length
        self.template = template
        copypercentage = 0.1
        mutatepercentage = 0.6
        wildcardpercentage = 0.3
        self.goobers = []
        
        for _ in range(int(goobercount * copypercentage)):
            self.goobers.append(Goober(0, template))
        for _ in range(int(goobercount * mutatepercentage)):
            self.goobers.append(Goober(1, template))
        for _ in range(int(goobercount * wildcardpercentage)):
            self.goobers.append(Goober(0, None))
    
    def runsim(self):
        for _ in range(self.length):
            alive = False
            for goober in self.goobers:
                if goober.state["health"] > 0:
                    goober.act(goober.decide())
                    alive = True
            if not alive:
                break
    
    def returnfittest(self):
        fittest = None
        fitnessscore = 0
        for goober in self.goobers:
            fitness = goober.returnfitness()
            if fitness > fitnessscore or (fitness == fitnessscore and random.randint(0, 1) == 1):
                fittest = goober
                fitnessscore = fitness

        #for i, goober in enumerate(self.goobers):
        #    print("goober", i + 1, "fitness:", goober.returnfitness(), "mutated" if goober.mutated == 1 else "copy", "highest" if goober == fittest else "", goober.template == self.template)

        return (fittest.template if fittest else fittest), fitnessscore, fittest