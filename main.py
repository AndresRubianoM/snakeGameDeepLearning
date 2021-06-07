import matplotlib.pyplot as plt
import numpy as np
import csv

from snakeGame.screen import Screen

from learningModels.process import LearningProcess, compute_avg_return, points_history
from learningModels.typesSnakeGame import SnakeAI, SnakeAIBorders

from learningModels.GameEnv import SnakeGameEnv
from utils import training_loops, sampler

def training():
    snake = SnakeAI([30,30])
    agent_environment = SnakeGameEnv(snake, -100)
    Lp = LearningProcess(agent_environment)
    Lp.pre_learning_process()
    Lp.training() 
    Lp.policy_saver()

def training_2():
    snake = SnakeAIBorders([30,30])
    rewards = {
        'aprox': [1,0],
        'eat':[10,0],
        'dead':[-100, 0]
    }
    agent_environment = SnakeGameEnv(snake, rewards, 33)
    Lp = LearningProcess(agent_environment)
    Lp.pre_learning_process()
    Lp.training() 
    Lp.policy_saver()

def training_3():
    snake = SnakeAIBorders([30,30])
    rewards = {
        'aprox': [1,0],
        'eat':[10,0],
        'dead':[-100, 0]
    }
    agent_environment = SnakeGameEnv(snake, rewards, 33)
    Lp = LearningProcess(agent_environment)
    Lp.pre_learning_process()
    returns, losses = Lp.training() 
    visualization(Lp, returns, losses)
    path = Lp.policy_saver()
    save_returns = path + '/returns.csv'
    save_losses = path + '/losses.csv'
    write_data(save_returns, returns)
    write_data(save_losses, losses)
    returns_read = read_data(save_returns)
    losses_read = read_data(save_losses)
    visualization(Lp, returns_read, losses_read)

def training_4():
    snakes_games = [SnakeAI([30,30]), SnakeAIBorders([30,30])]
    for snake in snakes_games:
        rewards = [{
            'aprox': [1,0],
            'eat':[10,0],
            'dead':[-100, 0]
            },
            {
            'aprox': [2,-1],
            'eat':[10,0],
            'dead':[-100, 0]
            },
        ]
        
        for reward in rewards:

            agent_environment = SnakeGameEnv(snake, reward, len(snakes_games[0].state()))
            for i in range(3):
                Lp = LearningProcess(agent_environment)
                Lp.pre_learning_process()
                returns, losses = Lp.training() 
                visualization(Lp, returns, losses)
                path = Lp.policy_saver(i)
                save_returns = path + '/returns.csv'
                save_losses = path + '/losses.csv'
                write_data(save_returns, returns)
                write_data(save_losses, losses)


def training_loops_2(snakes_games, rewards, redundance):
    for snake in snakes_games:
        for reward in rewards:
            agent_environment = SnakeGameEnv(snake, reward, len(snake.state()))
            Lp = LearningProcess(agent_environment)
            for i in range(redundance):
                print('*-*'*15)
                print('iteration {} using the reward {} with the rules/game {} and input {}'.format(i, reward, snake.__class__.__name__, len(snake.state())))
                Lp.pre_learning_process()
                returns, losses = Lp.training() 
                path = Lp.policy_saver(i)
                save_returns = path + '/returns.csv'
                save_losses = path + '/losses.csv'
                write_data(save_returns, returns)
                write_data(save_losses, losses)


def samples_1():
    snake = SnakeAI([30,30])
    rewards = {
        'aprox': [1,0],
        'eat':[10,0],
        'dead':[-100, 0]
    }
    agent_environment = SnakeGameEnv(snake, rewards, 33)
    Lp = LearningProcess(agent_environment)
    policy = Lp.load_previous_policy(0,0)
    return compute_avg_return(Lp.eval_env, policy, 30)


def samples_2(snake, reward, num_reward=0, iteration=0, num_episodes=30):
    
    agent_environment = SnakeGameEnv(snake, reward, len(snake.state()))
    Lp = LearningProcess(agent_environment)
    policy = Lp.load_previous_policy(num_reward, iteration)
    return compute_avg_return(Lp.eval_env, policy, num_episodes)

def samples_3(snake, reward, num_reward=0, iteration=0, num_episodes=30):
    
    agent_environment = SnakeGameEnv(snake, reward, len(snake.state()))
    Lp = LearningProcess(agent_environment)
    policy = Lp.load_previous_policy(num_reward, iteration)
    return points_history(Lp.sample_env, policy, num_episodes)

    

def playing_AI():
    snake = SnakeAI([30,30])
    agent_environment = SnakeGameEnv(snake)
    screen = Screen(300, 300, [30, 30], 5)
    Lp = LearningProcess(agent_environment)

    Lp.play_previous_policy(screen)

def write_data(file, data):
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

def visualization2(returns, losses):
    fig,ax = plt.subplots(1,2)
    fig.suptitle('Training process performance', fontsize=16)
    pass
    

def visualization(process, returns, losses):
    """"""
    fig, ax = plt.subplots(1, 2)
    #Title
    fig.suptitle('Training process performance', fontsize=16)
    #Data
    returns = np.transpose(np.matrix(returns))
    iterations_avg = range(0, process.num_iterations + 1, process.eval_interval)
    iterations_loss = range(0, process.num_iterations, process.log_interval)
    #Graph 1
    for i in range(len(returns)):
        ax[0].plot(iterations_avg, np.ravel(returns[i]), alpha = 0.15, color='gray')
    avg = np.mean(np.transpose(returns), axis=1)
    ax[0].plot(iterations_avg, avg, color='red')
    ax[0].set_ylabel('Average Return')
    ax[0].set_xlabel('Iterations')
    ax[0].set_title('Evolution of the Average Return ')
    #Graph 2
    ax[1].plot(iterations_loss, losses)
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Iterations')
    ax[1].set_title('Evolution of the loss')
    plt.show()
    

if __name__ == "__main__":
    #training()
    #training_2()
    #training_3()
    #training_4()
    #playing_AI()
    snakes_games = [SnakeAI([30,30]), SnakeAIBorders([30,30])]
    rewards = [{
            'aprox': [1,0],
            'eat':[10,0],
            'dead':[-100, 0]
            },
            {
            'aprox': [1,0],
            'eat':[10,0],
            'dead':[-10, 0]
            },
            {
            'aprox': [1,-1],
            'eat':[10,0],
            'dead':[-100, 0]
            },
            {
            'aprox': [3,-1],
            'eat':[10,0],
            'dead':[-100, 0]
            },
            {
            'aprox': [1,0],
            'eat':[20,0],
            'dead':[-10, 0]
            },
            {
            'aprox': [3,-1],
            'eat':[20,0],
            'dead':[-10, 0]
            },
    ]

 
    #training_loops(snakes_games, rewards,2)
    #print(samples_3(snakes_games[0], rewards[0], 0, 0,1))
    for i in range(50):
        a =sampler(snakes_games[0], rewards[0], 0, 0, 1, True)
        print(len(a), a[-1])


    
    #training_5(snakes_games, rewards, 1)
    
    
    
