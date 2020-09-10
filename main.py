from snakeGame.screen import Screen

from learningModels.process import LearningProcess
from learningModels.snake_AI import SnakeAI
from learningModels.GameEnv import SnakeGameEnv

def training():
    snake = SnakeAI([30,30])
    agent_environment = SnakeGameEnv(snake)
    Lp = LearningProcess(agent_environment)
    Lp.pre_learning_process()
    Lp.training() 
    Lp.policy_saver()


def playing_AI():
    snake = SnakeAI([30,30])
    agent_environment = SnakeGameEnv(snake)
    screen = Screen(300, 300, [30, 30], 5)
    Lp = LearningProcess(agent_environment)

    Lp.play_previous_policy(screen)
    




if __name__ == "__main__":
    training()
    playing_AI()
    

    
