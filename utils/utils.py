import matplotlib.pyplot as plt
import numpy as np
import csv

from learningModels.process import LearningProcess, compute_avg_return
from learningModels.GameEnv import SnakeGameEnv


def write_data(file, data):
    """Save the data in csv file"""
    
    with open(file, 'w') as f:
        data_writer = csv.writer(f, delimiter=',')
        for row in data:
            data_writer.writerow(row)


def read_data(file):
    data = []
    with open(file, 'r') as f:
        data_reader = csv.reader(f, delimiter=',')
        for row in data_reader:
            data.append([float(i) for i in row])
    
    return data


def training_loops(snakes_games, rewards, redundance):
    for snake in snakes_games:
        for num, reward in enumerate(rewards):
            agent_environment = SnakeGameEnv(snake, reward, len(snake.state()))
            Lp = LearningProcess(agent_environment)
            for i in range(redundance):
                print('*-*'*15)
                print('iteration {} using the reward {} with the rules/game {} and input {}'.format(i, reward, snake.__class__.__name__, len(snake.state())))
                Lp.pre_learning_process()
                returns, losses = Lp.training() 
                path = Lp.policy_saver(num, i)
                save_returns = path + '/returns.csv'
                save_losses = path + '/losses.csv'
                write_data(save_returns, returns)
                write_data(save_losses, losses)


def sampler(snake, reward, num_reward=0, iteration=0, num_episodes=30):
    
    agent_environment = SnakeGameEnv(snake, reward, len(snake.state()))
    Lp = LearningProcess(agent_environment)
    policy = Lp.load_previous_policy(num_reward, iteration)
    return compute_avg_return(Lp.sample_env, policy, num_episodes)
