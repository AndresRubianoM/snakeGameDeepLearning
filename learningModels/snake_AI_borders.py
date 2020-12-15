import copy
import numpy as np

#Snake base class for training
from learningModels.snake_AI import SnakeAI


class SnakeAIBorders(SnakeAI):
    """Snake class with defined borders for training."""

    def __init__(self, limits):
        super(SnakeAIBorders, self).__init__(limits)
        self.border_out = False
        
    
    def displacement(self):
        """Pass the change of velocity to move the snake and rules of displacement
        in the screen (In this class the screen have defined borders that cause dead 
        to the snake)"""

        #Applied the move to the snake head
        self.body[0].velocities = self.velocities
        self.body[0].make_move() 

        #Define if the head touch the borders
        if self.body[0].position[0] > self.limits[0] - 1:
            self.body[0].position[0] -= 1
            self.border_out = True
        if self.body[0].position[0] < 0:
            self.body[0].position[0] = 0
            self.border_out = True
        if self.body[0].position[1] > self.limits[1] - 1:
            self.body[0].position[1] -= 1
            self.border_out = True
        if self.body[0].position[1] < 0:
            self.body[0].position[1] = 0
            self.border_out = True

        #Save the information necesary for each part of the snake plus one
        self.previous_data.append(copy.deepcopy(self.body[0].position))
        self.previous_data = self.previous_data[-(len(self.body) + 1):]
    
        for i, snake_part in enumerate(self.body[1:]):
            snake_part.position = self.previous_data[-(i + 2)]
        
        #self.dead_condition()


    def dead_condition(self):
        """The snake will die if the head touch the borders"""
            
        if super().dead_condition() or self.border_out:
            return True
        else:
            return False
            
    
    def _reset_game(self):
        super()._reset_game()
        self.border_out = False


    def obstacles(self):

        """Function that looks the objects arround the head of the snake, 
        and build a sub matrix where 1 represents an object and 0 the aussence of one.
        In this case there are borders so that are marks as an object.
        """

        #Radious arround the head
        limit_sight = 2
        head = self.body[0].position
        binary_map_complete = self.complete_mapping()
        map_matrix = np.matrix(binary_map_complete)
        obstacles = []

        #limits in all directions
        left_x = head[0] - limit_sight
        right_x = head[0] + limit_sight
        up_y = head[1] - limit_sight
        down_y = head[1] + limit_sight

        #submatrix with limits size
        snake_sight = map_matrix[up_y:down_y+1, left_x:right_x+1]


        #Special cases where the snake approximates to the borders
        ##Corners
        if left_x < 0 and up_y < 0:
            snake_sight = map_matrix[0:down_y+1, 0:right_x+1]
            interval_x = [self.limits[0] + left_x, self.limits[0]]
            interval_y = [self.limits[1] + up_y, self.limits[1]]
            interval_x_matrix = np.ones([(down_y + 1), interval_x[1] - interval_x[0]], dtype=int)
            interval_y_matrix = np.ones([interval_y[1] - interval_y[0], right_x + 1 - left_x], dtype=int)
            snake_sight = np.c_[interval_x_matrix, snake_sight]
            snake_sight = np.r_[interval_y_matrix, snake_sight]  
            return snake_sight
        
        if left_x < 0 and down_y > self.limits[1] - 1:
            snake_sight = map_matrix[up_y:self.limits[1], 0:right_x+1]
            interval_x = [self.limits[0] + left_x, self.limits[0]]
            interval_y = [0, down_y - self.limits[1] + 1]
            interval_x_matrix = np.ones([(self.limits[1] - up_y), interval_x[1] - interval_x[0]], dtype=int)
            interval_y_matrix = np.ones([interval_y[1] - interval_y[0], right_x + 1 - left_x], dtype=int)
            snake_sight = np.c_[interval_x_matrix, snake_sight]
            snake_sight = np.r_[snake_sight, interval_y_matrix]
            return snake_sight
        
        if right_x > self.limits[0]-1 and up_y < 0:
            snake_sight = map_matrix[0:down_y+1, left_x:self.limits[0]]
            interval_x = [0, right_x - self.limits[0] + 1]
            interval_y = [self.limits[1] + up_y, self.limits[1]]
            interval_x_matrix = np.ones([(down_y + 1), interval_x[1] - interval_x[0]], dtype=int)
            interval_y_matrix = np.ones([interval_y[1] - interval_y[0], right_x + 1 - left_x], dtype=int)
            snake_sight = np.c_[snake_sight, interval_x_matrix]
            snake_sight = np.r_[interval_y_matrix, snake_sight]
            return snake_sight
        
        if right_x > self.limits[0]-1 and down_y > self.limits[1]-1:
            snake_sight = map_matrix[up_y:self.limits[1], left_x:self.limits[0]]
            interval_x = [0, right_x - self.limits[0] + 1]
            interval_y = [0, down_y - self.limits[1] + 1]
            interval_x_matrix = np.ones([(self.limits[1] - up_y), interval_x[1] - interval_x[0]], dtype=int)
            interval_y_matrix = np.ones([interval_y[1] - interval_y[0], right_x + 1 - left_x], dtype=int)
            snake_sight = np.c_[snake_sight, interval_x_matrix]
            snake_sight = np.r_[snake_sight, interval_y_matrix]
            return snake_sight

        ##Middle
        if left_x < 0:
            snake_sight = map_matrix[up_y:down_y+1, 0:right_x+1]
            interval_x = [self.limits[0] + left_x, self.limits[0]]
            interval_x_matrix = np.ones([(down_y + 1 - up_y), interval_x[1] - interval_x[0]], dtype=int)
            snake_sight = np.c_[interval_x_matrix, snake_sight]
            return snake_sight

        if right_x > self.limits[0]-1:
            snake_sight = map_matrix[up_y:down_y+1, left_x:self.limits[0]]
            interval_x = [0, right_x - self.limits[0] + 1]
            interval_x_matrix = np.ones([(down_y + 1 - up_y), interval_x[1] - interval_x[0]], dtype=int)
            snake_sight = np.c_[snake_sight, interval_x_matrix]
            return snake_sight

        if up_y < 0:
            snake_sight = map_matrix[0:down_y+1, left_x:right_x+1]
            interval_y = [self.limits[1] + up_y, self.limits[1]]
            interval_y_matrix = np.ones([interval_y[1] - interval_y[0], right_x + 1 - left_x], dtype=int)
            snake_sight = np.r_[interval_y_matrix, snake_sight]
            return snake_sight
   
        if down_y > self.limits[1]-1:
            snake_sight = map_matrix[up_y:self.limits[1], left_x:right_x+1]
            interval_y = [0, down_y - self.limits[1] + 1]
            interval_y_matrix = np.ones([interval_y[1] - interval_y[0], right_x + 1 - left_x], dtype=int)
            snake_sight = np.r_[snake_sight, interval_y_matrix]
            return snake_sight

        return snake_sight

