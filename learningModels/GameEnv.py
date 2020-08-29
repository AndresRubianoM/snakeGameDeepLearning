import tensorflow as tf 
import numpy as np
#Snake Game class
from learningModels.snake_AI import SnakeAI
#Reinforce learning
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class SnakeGameEnv(py_environment.PyEnvironment):
    """Enviroment of the snake game"""

    def __init__(self):
        #Object of snake Game
        limits = [10,10]
        self.snake = SnakeAI(limits)
        #Shape and limits of the possible actions
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3)
        #Shape and limits of the observations (not the state)
        #self._observation_spec = array_spec.BoundedArraySpec(
        #   shape=(limits[0]*limits[1],), dtype=np.int32, minimum=0.0, maximum=1.0, name='observation')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(12, ), dtype=np.int32, minimum=0, name='observation')
        
        self._state = self.snake.state() #complete_map.ravel()
        self._episode_ended = False


    def action_spec(self):
        return self._action_spec


    def observation_spec(self):
        return self._observation_spec

    
    def _reset(self):
        self.snake._reset_game()
        self._state = self.snake.state() #.complete_map.ravel()
        self._episode_ended = False
        return ts.restart(self._state)#np.array([self._state], dtype=np.int32))


    def _step(self, action):

        if self.snake.dead_condition():
            return self.reset()

        if action in [0,1,2,3]:
            self.snake.movement(action)
            self._state = self.snake.state() #.complete_mapping().ravel()
        else:
            raise ValueError('action must be 0,1,2 or 3 and was {}'.format(action))

        reward = 0   
        reward += 10 if self.snake.eat() else 0 
        reward += 1 if self.snake.near_way() else -1
        
        if self.snake.dead_condition():
            reward += -100
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount=0.95)
            

 