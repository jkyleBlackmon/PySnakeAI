import torch
import random
import numpy as np
import time
from collections import deque # store game memory
from game import SnakeGameAI, Direction, Point

from model1 import Linear_QNet, QTrainer
from plotter import plot

MAX_MEMORY = 10_000    # Memory: stores 100,000 items
BATCH_SIZE = 100       # Batch Size: How much per run
LR = 0.001              # Learning Rate: How fast model learns


class Agent:

    '''
    Intializes the PyTorch Agent with default 0s
    '''
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0    # controls randomness
        self.gamma = 0.9      # discount rate (Must be < 1 but can play around with it)
        self.memory = deque(maxlen=MAX_MEMORY) # if memory is exceeded, deletes old memory automatically
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger Left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            game.food.x < game.head.x, # Food LEFT
            game.food.x > game.head.x, # Food RIGHT
            game.food.y < game.head.y, # Food UP
            game.food.y > game.head.y, # Food DOWN
        ]
        
        return np.array(state, dtype=int) # Format and convert the state data

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # if > MAX_MEMORY, pop.left


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # random sample
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Get BATCH_SIZE num of samples
        else:
            mini_sample = self.memory

        # states, actions, rewards, next_states, dones = zip(*mini_sample)    # Look into zip function
        # self.trainer.train_step(states, actions, rewards, next_states, dones)
        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # random moves: tradeoff btwn exploration/exploitation
        # Over time we want to explore less and exploit more
        self.epsilon = 80 - self.n_games 
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # raw value prediction, take max and set index
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    print("Training ... [ Game", agent.n_games + 1, "]")
    while True:
        start_time = time.time()
        # get old/current state
        old_state = agent.get_state(game)
        # get move based on curr state
        final_move = agent.get_action(old_state)
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # print((old_state, new_state, reward, done, score))
        # train short memory
        print("Training Short Memory... ")
        agent.train_short_memory(old_state, final_move, reward, new_state, done)
        print("Done.")

        # remember
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Training ... [ Game {agent.n_games} ] - Time taken: {elapsed_time:.2f} seconds")
            print(f"Game {agent.n_games} Score {score} Record {record}")

            # Save the model every 10 games
            if agent.n_games % 10 == 0:
                agent.model.save()

            # Break after a certain number of games to prevent infinite loop
            if agent.n_games > 1000:  # Adjust as needed
                break

            # TODO: plot result
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)

            print("Plotted Game", agent.n_games)


if __name__ == '__main__':
    train()