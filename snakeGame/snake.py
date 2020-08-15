import random
import copy

import pygame

class Square:
    """Square of the size defined by the columns and rows
    position = list [x,y]
    velocities = list [vel_x, vel_y]
    color = rgb tuple (R, G, B)
    
    """
    def __init__(self, position, color = (255,255,255)):
        self.color = color
        self.position = position
        self.velocities = [0,0]

        
    def draw(self, screen, rows, columns):
        pygame.draw.rect(screen, self.color, (self.position[0] * columns + 1, self.position[1] * rows + 1, rows - 1, columns - 1))
    

    def make_move(self):
        self.position[0] = self.position[0] + self.velocities[0]
        self.position[1] = self.position[1] + self.velocities[1]
    


class Prey(Square):
    """Objective of the snake to point"""
    def __init__(self, limits):
        self.limits = limits
        super().__init__(self._random_position(), (255,0,0))
    
    
    def relocation(self, snake_parts):
        prey_position = self._random_position()
        parts = [ part.position for part in snake_parts]
        while prey_position in parts:
            prey_position = self._random_position()

        self.position = prey_position
    

    def _random_position(self):
        return [random.randrange(self.limits[0] - 1), random.randrange(self.limits[1] - 1)]


class Snake:
    """Snake contain all the information related with the body of the 
    snake in the game.
    position = list [x,y]
    limits = list[max_x, max_y]
    """

    def __init__(self, position, limits):
        self.position = position
        self.limits = limits

        self.body = [Square(position)]
        self.velocities = [0,0]
        
        self.punctuation = 0
        self.previous_data = []

    
    def movement(self, event):
        """Possible movements of the snake"""
        #Restrain the movement only to perpendicular ones
        if event.type == pygame.KEYDOWN:
            #if its moving horizontally only can move vertically in the next move
            if self.velocities[1] == 0:
                if event.key == pygame.K_UP:
                    self.velocities[0] = 0
                    self.velocities[1] = -1
                if event.key == pygame.K_DOWN:
                    self.velocities[0] = 0
                    self.velocities[1] = 1

            #if its moving vertically only can move horizontally in the next move
            if self.velocities[0] == 0:
                if event.key == pygame.K_LEFT:
                    self.velocities[0] = -1
                    self.velocities[1] = 0
                if event.key == pygame.K_RIGHT:
                    self.velocities[0] = 1
                    self.velocities[1] = 0


    def displacement(self):
        "Pass the change of velocity to move and rules of displacement"
        #Applied the move to the snake head
        self.body[0].velocities = self.velocities
        self.body[0].make_move() 

        #Transportation if passes the limits
        if self.body[0].position[0] > self.limits[0] - 1:
            self.body[0].position[0] = 0
        if self.body[0].position[0] < 0:
            self.body[0].position[0] = self.limits[0] - 1
        if self.body[0].position[1] > self.limits[1] - 1:
            self.body[0].position[1] = 0
        if self.body[0].position[1] < 0:
            self.body[0].position[1] = self.limits[1] - 1
    
        #Save the information necesary for each part of the snake plus one
        self.previous_data.append(copy.deepcopy(self.body[0].position))
        self.previous_data = self.previous_data[-(len(self.body) + 1):]
    
        for i, snake_part in enumerate(self.body[1:]):
            snake_part.position = self.previous_data[-(i + 2)]
        
        self.dead_condition()
        
          
    def eat(self, prey):
        if prey.position == self.body[0].position:
            self.punctuation += 1
            prey.relocation(self.body)
            self.grow()
    

    def grow(self):
        self.body.append(Square(copy.deepcopy(self.previous_data[-(self.punctuation + 1)])))
    

    def dead_condition(self):
        head = self.body[0].position
        body = [part.position for part in self.body[1:]]
        print(head, body)
        if head in body:
            self.body = [self.body[0]]
            self.body[0].position = [(self.limits[0]//2), (self.limits[1]//2)]

            self.punctuation = 0 

             
        
    def draw(self, screen, rows, columns):
        for square in self.body:
            square.draw(screen, rows, columns)


    
    
    



