from snakeGame.screen import Screen


game = Screen(300, 300, [15,15], 5)
game.start_pygame()
game.game_cycle()