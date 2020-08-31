import pygame
#local Imports
from .snake import Prey
from .snake import Snake


class Screen:
	"""Manage the pygame screen with its properties"""
	
	def __init__(self, width, height, number_spaces, frame_rate):
		self.width      = width
		self.height     = height 
		self.number_spaces = number_spaces
		self.frame_rate = frame_rate 

		self.rows    = self.width // self.number_spaces[0]
		self.columns = self.height // self.number_spaces[1]

		self.running = True


	def start_pygame(self):
		"""Initialize the pygame objects to run the screen"""

		#Instance the pygame objects 
		pygame.init()
		self.screen = pygame.display.set_mode([self.width, self.height])
		pygame.display.set_caption('Game Window')		
		#Control the frame rate 
		self.clock = pygame.time.Clock()


	def update_window_frame(self, game_items):
		"""Function that draw the components in the screen"""
		#The Game is design with discretes steps 

		#Black backgorund
		self.screen.fill((0,0,0))
			
		#Lines 
		for point in range(self.rows, self.height, self.rows):
			pygame.draw.line(self.screen, (255, 255, 255), (0, point), (self.width, point))

		for point in range(self.columns, self.width, self.columns):
			pygame.draw.line(self.screen, (255, 255, 255), (point, 0), (point, self.height))

		#Draw all the items of the game	
		for item in game_items.values():
			for objects in item.values():
				objects.draw(self.screen, self.rows, self.columns)
		
		pygame.display.update()

	
	def events_handler(self, game_items):
		"""Manage the evetns to execute the game items actions"""

		#Exit of the screen game with the x button
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				exit()
			
			#Manage the movements capabilities of the mobile items
			for item in game_items['mobile_items'].values():
				item.movement(event)
		
		#In each frame continues the previous movement
		for item in game_items['mobile_items'].values():
			item.displacement()
		

						
	def game_cycle(self, items):
		"""Infinite loop while the game is executing"""
		
		while self.running:
			pygame.event.get()
			dt = self.clock.tick(self.frame_rate)
			
			self.update_window_frame(items)
			self.events_handler(items)
			items['mobile_items']['Snake'].eat(items['static_items']['Prey'])
			
				
			


	







		
		
