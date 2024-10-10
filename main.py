import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import duckdb
import io
import json

db = duckdb.connect("world.db")

db.sql("CREATE TABLE IF NOT EXISTS people (id INT UNIQUE NOT NULL, name VARCHAR, hunger INT, fatigue INT, money INT);")
db.sql("CREATE TABLE IF NOT EXISTS models (id INT UNIQUE NOT NULL, model JSON, FOREIGN KEY (id) REFERENCES people (id));")

try:
    db.sql("INSERT INTO people (id, name, hunger, fatigue, money) VALUES (1, 'Vladick', 0, 0, 100);")
except:
    pass

# Define a simple neural network for policy or Q-value estimation
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.fc(x)

# RL Algorithm class
class ReinforcementLearner:
    def __init__(self, input_size, output_size, lr=1e-3):
        self.model = NN(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()  # For Q-Learning
        self.entropy_coefficient = 0.01
        
    def select_action(self, state):
        """Convert state to tensor, get action from NN"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_logits = self.model(state_tensor)
        action_probs = torch.softmax(action_logits, dim=1)

        action = torch.multinomial(action_probs, num_samples=1).item()
        return action
    
    def update(self, state, action, reward, next_state):
        """Update the model based on state-action-reward-next state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        predicted_q_value = self.model(state_tensor)[0, action]
        
        # Assuming max(Q(next_state, all_actions)) for Q-learning
        target_q_value = reward + 0.99 * torch.max(self.model(next_state_tensor)).item()
        
        loss = self.criterion(predicted_q_value, torch.FloatTensor([target_q_value]))

        # Add entropy term to encourage exploration
        action_logits = self.model(state_tensor)
        action_probs = torch.softmax(action_logits, dim=-1)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))  # Entropy term

        total_loss = loss - self.entropy_coefficient * entropy  # Add entropy penalty
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def save_model(self, id):
        """Serialize and save the model to the database in JSON format."""
        model_params = {k: v.tolist() for k, v in self.model.state_dict().items()}
        model_json = json.dumps(model_params)

        if db.sql(f"SELECT * FROM models WHERE id = {id}").fetchone():
            db.sql(f"UPDATE models SET model = '{model_json}' WHERE id = {id};")
        else:
            db.sql(f"INSERT INTO models (id, model) VALUES ({id}, '{model_json}');")

    def load_model(self, id):
        """Load the model from the database if it exists."""
        model_json = db.sql(f"SELECT model FROM models WHERE id = {id}").fetchone()
        if model_json:
            model_params = json.loads(model_json[0])
            model_state_dict = {k: torch.FloatTensor(np.array(v)) for k, v in model_params.items()}
            self.model.load_state_dict(model_state_dict)

# Simulating a custom environment
class Environment:
    def __init__(self):
        self.state = None
    
    def step(self, id, action):
        self.state = np.array(db.sql(f"SELECT hunger, fatigue, money FROM people WHERE id = {id}").fetchone()).flatten()
        hunger, fatigue, money = self.state
        reward = 0

        if action == 0:  # sleep
            if fatigue < 10:
                reward += -100
            else:
                reward += max(0, 70 - hunger)
            fatigue = max(fatigue - 15, 0)
            hunger = min(hunger + 3, 100)

        elif action == 1:  # eat
            if hunger < 15:
                reward += -100
            else:
                reward += max(0, 70 - fatigue)
            fatigue = min(fatigue + 4, 100)
            if money >= 5:
                hunger = max(hunger - 20, 0)
                money = max(money - 5, 0)
            else:
                reward += -100

        elif action == 2:  # work
            if fatigue > 90 or hunger > 90:
                reward -= hunger/2 + fatigue
            elif money < 30:
                reward += max(0 , 100 - fatigue/4 - hunger/6)
            money += int(20 - max(0, fatigue/10) - max(0, hunger/10))
            fatigue = min(fatigue + 7, 100)
            hunger = min(hunger + 6, 100)

        elif action == 3:  # entertainment
            if money >= 25:
                reward += max(0, int(100 - fatigue/2 - hunger/4))
                money = max(money - 25, 0)  # Entertainment costs more than food
            else:
                reward = -100
            fatigue = min(fatigue + 6, 100)
            hunger = min(hunger + 5, 100)
        
        next_state = np.array([hunger, fatigue, money])
        db.sql(f"UPDATE people SET hunger = {hunger}, fatigue = {fatigue}, money = {money} WHERE id = {id};")

        return next_state, reward

# Example of using the learner
env = Environment()

printloop = 480
sleeps = 0
eats = 0
works = 0
entertainments = 0
rewards = 0

while True:
    for id in np.array(db.sql("SELECT id FROM people").fetchall()).flatten():

        learner = ReinforcementLearner(input_size=3, output_size=4)
        learner.load_model(id)

        state = np.array(db.sql(f"SELECT hunger, fatigue, money FROM people WHERE id = {id}").fetchone()).flatten()
        action = learner.select_action(state)
        
        # Simulate environment step
        next_state, reward = env.step(id, action)
        
        # Update NN with the transition
        learner.update(state, action, reward, next_state)
        learner.save_model(id)
        db.commit()
        
        # Move to the next state
        env.state = next_state

        if action == 0:
                sleeps += 1
        elif action == 1:
            eats += 1
        elif action == 2:
            works += 1
        elif action == 3:
            entertainments += 1
        rewards += reward

        if printloop > 0:
            printloop -= 1
        else:
            printloop = 480
            print(f"per day:\nhours of sleep: {sleeps/20}\nmeals: {eats/20}\nwork hours: {works/20}\nentertainment hours: {entertainments/20}\ndaily average reward: {rewards/20}\n")
            sleeps = 0
            eats = 0
            works = 0
            entertainments = 0
            rewards = 0