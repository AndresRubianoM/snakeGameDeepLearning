import numpy as np
#Game logic classes
from snakeGame import Snake, Prey, Screen


class SnakeAI(Snake):
    """Snake class child, adapted to be part of the environment of 
    tf-agents framework.

    The snakeAI class envolves all the items of the snake game (the snake and 
    the prey), that are contain in the predefined limits.

    limits: [x,y] ints 

    To train the game it was decided to pass a square sub-matrix with the 'sight' of the snake 
    given a state, the size will be n in each direction from the position of player so the total 
    size is 2n + 1.

    sight (int): distance from the snake's head.
    """

    def __init__(self, limits, sight=2):
        super(SnakeAI, self).__init__(limits)
        self.prey = Prey(self.limits)
        self.snake_sight = sight
        self.complete_map = np.zeros(limits, dtype=np.float32)
    
    
    def movement(self, action):
        """Changes the direction of snake's movement
        UP: 0    LEFT: 2
        DOWN: 1  RIGHT: 3
        """

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
        
        self.displacement()


    def eat(self):
        "Condition to win points 'eating the prey'"

        if self.prey.position == self.body[0].position:
            self.punctuation += 1
            self.prey.relocation(self.body)
            self.grow()
            return True
        else:
            return False


    def complete_mapping(self):
        """Binary map of the full game. Mark the objects in the matrix of posible positions,
        1 if its filled and 0 if it not."""

        self._reset_map()
        #position_prey = self.prey.position
        #self.complete_map[position_prey[1], position_prey[0]] = 1.0
        position_body = [part.position for part in self.body]

        for position in position_body:
            self.complete_map[position[1], position[0]] = 1

        return self.complete_map


    def state(self):
        """State for the learning process, returns a binary array, 
        1 if the condition is true 0 if not."""

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
        obstacles = np.ravel(self.obstacles())
        return (np.concatenate((
            [prescence_prey_right,
            prescence_prey_left,
            prescence_prey_up,
            prescence_prey_down,
            actual_direction_right,
            actual_direction_left,
            actual_direction_up,
            actual_direction_down],
            obstacles
            )))

        
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

    
    def obstacles(self):

        """Function that looks the objects arround the head of the snake, 
        and build a sub matrix where 1 represents an object and 0 the aussence of one.
        In this case there are not borders, so the snake look to the opposite site.
        """

        #Radious arround the head
        limit_sight = self.snake_sight
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
            interval_x_matrix = map_matrix[0:down_y+1, interval_x[0]:interval_x[1]]
            interval_y_matrix = map_matrix[interval_y[0]:interval_y[1], 0:right_x+1]
            interval_x_y_matrix = map_matrix[interval_y[0]:interval_y[1], interval_x[0]:interval_x[1]]
            temporal = np.c_[interval_x_y_matrix, interval_y_matrix]
            snake_sight = np.c_[interval_x_matrix, snake_sight]
            snake_sight = np.r_[temporal, snake_sight]  
            return snake_sight
        
        if left_x < 0 and down_y > self.limits[1] - 1:
            snake_sight = map_matrix[up_y:self.limits[1], 0:right_x+1]
            interval_x = [self.limits[0] + left_x, self.limits[0]]
            interval_y = [0, down_y - self.limits[1] + 1]
            interval_x_matrix = map_matrix[up_y:self.limits[1], interval_x[0]:interval_x[1]]
            interval_y_matrix = map_matrix[interval_y[0]:interval_y[1], 0:right_x+1]
            interval_x_y_matrix = map_matrix[interval_y[0]:interval_y[1], interval_x[0]:interval_x[1]]
            temporal = np.c_[interval_x_y_matrix, interval_y_matrix]
            snake_sight = np.c_[interval_x_matrix, snake_sight]
            snake_sight = np.r_[snake_sight, temporal]
            return snake_sight
        
        if right_x > self.limits[0]-1 and up_y < 0:
            snake_sight = map_matrix[0:down_y+1, left_x:self.limits[0]]
            interval_x = [0, right_x - self.limits[0] + 1]
            interval_y = [self.limits[1] + up_y, self.limits[1]]
            interval_x_matrix = map_matrix[0:down_y+1, interval_x[0]:interval_x[1]]
            interval_y_matrix = map_matrix[interval_y[0]:interval_y[1], left_x:self.limits[0]]
            interval_x_y_matrix = map_matrix[interval_y[0]:interval_y[1], interval_x[0]:interval_x[1]]
            temporal = np.c_[interval_y_matrix, interval_x_y_matrix]
            snake_sight = np.c_[snake_sight, interval_x_matrix]
            snake_sight = np.r_[temporal, snake_sight]
            return snake_sight
        
        if right_x > self.limits[0]-1 and down_y > self.limits[1]-1:
            snake_sight = map_matrix[up_y:self.limits[1], left_x:self.limits[0]]
            interval_x = [0, right_x - self.limits[0] + 1]
            interval_y = [0, down_y - self.limits[1] + 1]
            interval_x_matrix = map_matrix[up_y:self.limits[1], interval_x[0]:interval_x[1]]
            interval_y_matrix = map_matrix[interval_y[0]:interval_y[1], left_x:self.limits[0]]
            interval_x_y_matrix = map_matrix[interval_y[0]:interval_y[1], interval_x[0]:interval_x[1]]
            temporal = np.c_[interval_y_matrix, interval_x_y_matrix]
            snake_sight = np.c_[snake_sight, interval_x_matrix]
            snake_sight = np.r_[snake_sight, temporal]
            return snake_sight

        ##Middle
        if left_x < 0:
            snake_sight = map_matrix[up_y:down_y+1, 0:right_x+1]
            interval_x = [self.limits[0] + left_x, self.limits[0]]
            interval_x_matrix = map_matrix[up_y:down_y+1, interval_x[0]:interval_x[1]]
            snake_sight = np.c_[interval_x_matrix, snake_sight]
            return snake_sight

        if right_x > self.limits[0]-1:
            snake_sight = map_matrix[up_y:down_y+1, left_x:self.limits[0]]
            interval_x = [0, right_x - self.limits[0] + 1]
            interval_x_matrix = map_matrix[up_y:down_y+1, interval_x[0]:interval_x[1]]
            snake_sight = np.c_[snake_sight, interval_x_matrix]
            return snake_sight

        if up_y < 0:
            snake_sight = map_matrix[0:down_y+1, left_x:right_x+1]
            interval_y = [self.limits[1] + up_y, self.limits[1]]
            interval_y_matrix = map_matrix[interval_y[0]:interval_y[1], left_x:right_x+1]
            snake_sight = np.r_[interval_y_matrix, snake_sight]
            return snake_sight
   
        if down_y > self.limits[1]-1:
            snake_sight = map_matrix[up_y:self.limits[1], left_x:right_x+1]
            interval_y = [0, down_y - self.limits[1] + 1]
            interval_y_matrix = map_matrix[interval_y[0]:interval_y[1], left_x:right_x+1]
            snake_sight = np.r_[snake_sight, interval_y_matrix]
            return snake_sight

        return snake_sight
   
        
    def dead_condition(self):
        if super().dead_condition():
            self._reset_map()
            return True
        else:
            return False

    
    def near_way(self):
        """Evaluate if the snake's head is approaching to the prey
        position or going away."""

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

        


    
        



