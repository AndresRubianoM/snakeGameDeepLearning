import copy

from learningModels.snake_AI import SnakeAI


class SnakeAIBorders(SnakeAI):
    """Snake class with defined borders"""

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
        if (self.body[0].position[0] > self.limits[0] - 1) or (self.body[0].position[0] < 0) or (self.body[0].position[1] > self.limits[1] - 1) or (self.body[0].position[1] < 0):
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

