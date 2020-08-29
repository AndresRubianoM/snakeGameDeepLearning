import numpy as np
#Game logic classes
from snakeGame.snake import Snake
from snakeGame.snake import Prey
#Pygame screen
from snakeGame.screen import Screen


class SnakeAI(Snake):
    """Snake class prepare for the enviroment"""

    def __init__(self, limits):
        super(SnakeAI, self).__init__(limits)
        self.prey = Prey(self.limits)
        self.complete_map = np.zeros(limits, dtype=np.float32)
    
    def movement(self, action):
        #UP: 0    LEFT: 2
        #DOWN: 1  RIGHT: 3

        #if its moving horizontally only can move vertically in the next move
        if self.velocities[1] == 0:
            if action == 0 :
                self.velocities[0] = 0
                self.velocities[1] = -1
            if action == 1 :
                self.velocities[0] = 0
                self.velocities[1] = 1

        #if its moving vertically only can move horizontally in the next move
        if self.velocities[0] == 0:
            if action == 2 :
                self.velocities[0] = -1
                self.velocities[1] = 0
            if action == 3 :
                self.velocities[0] = 1
                self.velocities[1] = 0
        
        #execute displacement
        #print('before: ', self.body[0].position)
        self.displacement()
        #print('after: ', self.body[0].position)

    def eat(self):
        if self.prey.position == self.body[0].position:
            self.punctuation += 1
            self.prey.relocation(self.body)
            self.grow()
            return True
        else:
            return False


    def complete_mapping(self):
        self._reset_map()
        position_prey = self.prey.position
        position_body = [part.position for part in self.body]
        self.complete_map[position_prey[1], position_prey[0]] = 1.0

        for position in position_body:
            self.complete_map[position[1], position[0]] = 1.0

        return self.complete_map


    def state(self):
        #Mark in wich direction is the prey
        prescence_prey_right = 1 if (self.prey.position[0] > self.body[0].position[0]) else 0
        prescence_prey_left = 1 if (self.prey.position[0] < self.body[0].position[0]) else 0
        prescence_prey_up = 1 if (self.prey.position[1] < self.body[0].position[1]) else 0
        prescence_prey_down = 1 if (self.prey.position[1] > self.body[0].position[1]) else 0
        #Direction where is moving
        actual_direction_right = 1 if (self.velocities[0] == 1) else 0
        actual_direction_left =  1 if (self.velocities[0] == -1) else 0
        actual_direction_up = 1 if (self.velocities[1] == -1) else 0
        actual_direction_down = 1 if (self.velocities[1] == 1) else 0
        #Mark if is an obstacle
        body = [part.position for part in self.body]
        body_x = [part[0] for part in body]
        body_y = [part[1] for part in body]
        obstacle_right = 1 if ((self.body[0].position[0] + 1) in body_x) else 0
        obstacle_left = 1 if ((self.body[0].position[0] - 1) in body_x) else 0
        obstacle_up = 1 if ((self.body[0].position[1] - 1) in body_y) else 0
        obstacle_down = 1 if ((self.body[0].position[1] + 1) in body_y) else 0
        #return np.array([self.prey.position[0], self.prey.position[1], self.body[0].position[0], self.body[0].position[1]])
        return np.array([
            prescence_prey_right,
            prescence_prey_left,
            prescence_prey_up,
            prescence_prey_down,
            actual_direction_right,
            actual_direction_left,
            actual_direction_up,
            actual_direction_down,
            obstacle_right,
            obstacle_left,
            obstacle_up,
            obstacle_down,])


    def dead_condition(self):
        if super().dead_condition():
            self._reset_map()
            return True
        else:
            return False

    
    def near_way(self):
        prey_position = np.array(self.prey.position)
        actual_position = np.array(self.previous_data[-1])
        previous_position = np.array(self.previous_data[-2])

        difference_actual = np.linalg.norm(prey_position - actual_position)
        difference_previous = np.linalg.norm(prey_position - previous_position)

        if difference_actual < difference_previous:
            return True
        else:
            return False
        

    def _reset_map(self):
        self.complete_map = np.zeros(self.limits, dtype=int) 


    def _reset_game(self):
        super()._reset_game()
        self.prey.relocation(self.body)
        self._reset_map()

        


    
        



