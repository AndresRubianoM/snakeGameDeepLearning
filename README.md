# A more real approximation to the use of Reinforce Learning Models

Is common to find tutorials about how to build a reinforce learning model, but the majority of these tutorials focus on how to build a single model and not on how to analyze and build an useful tool. With this in mind, I want to show the analysis to define a better model and what implies analyzing a simple problem more deeply.

My priority is to show the analysis and process of building useful reinforced learning (RL) model and study how little changes in the parameters affect the result of the model, for that reason I chose to use the most common tools that are in different resources to learn this type of model:

- RL: Deep-Q network
- Problem/Game: Snake Game

# Context 

Before starting the analysis of the problem I must explain a little more what characteristics have both the model and the game.

Let's start with the model, As I mentioned I'll be using the Deep-Q network. the model's theory isn't too difficult but could be a little tricky at a first glance. 

## Library

To simplify the process I chose to use a specialized library called **Tensorflow Agents**, this library helps us to implement, test and deploy the RL models, furthermore it's flexible enough to modify each part of the process to our necessities.

I'm not going to explain in detail how to use all the possibilities of the library, but essentially only requires to use/extend the PyEnvironment class to define the "rules" of the game and configure the model itself (algorithm, architecture, memory, hyperparameters, training, testing). If someone is interested, should look at the official documentation which it is very complete with examples using different configurations.

## Game

Why the snake game? well, the answer is very simple, it's a very common game that most parts of people played at least one time in their life, and is easy to program it from scratch.

The rules of the game are easy but it's important to clarify specifically how are defined.

- The movement space of the game is discretized in NxN squares
- We only move the snakes head, the rest of the body follows it
- to get one point the head of the snake needs to be in the same position as the "prey"
- The prey is randomly located in an empty square
- The snake grows one square each time the player makes one point
- The snake/player lose when the head crash with some other part of its body

Finally, there is a special rule that generates two "different games" and is related to the borders. In one of the games, the snake will die if the head touch the border meanwhile in the other the snake will teleport to the opposite side. This change is very important when we analyze the behavior of the models because in one of the games the snake only has one way of losing when it makes a bad movement having at least 4 points, and in the other game, the snake could lose from the beginning.

## Process 

As stated before the model I used is the Deep-Q network, to use this model we need to define a few things:

- input variables: Since the space of movement is discretized we can represent everything as a binary map where 1 means that the square is occupied with something and 0 means that the square is empty,
also, we need to include a little more information about the situation of the player like in which direction is the prey and in which direction is the player moving.

- output: Is simple to answer this because there are only 4 types of movements (up, down, left, or right)

- number of neurons and number of hidden layers: To be strictly defining the number of neurons and hidden layers is a complete process that implies testing different configurations and checks the effectiveness of every each of them, but if we include these new possibilities into the analysis the combinations of the parameters increases a lot more of the necessary and doesn't contribute to the purpose of the analysis.

- Game policies: As you may know an important thing that we need to define in a RL model is the game policies which tell the algorithm what is a good movement and must receive a reward or a bad movement and must receive a punishment. Defining the policies depends on the behavior that we want the snake to accomplish, since the game itself is easy we only have to define a few items, specifically when the snake eats, approach to the prey and when it die.

Although I already named the general description of the parameters/data we need to define, it's necessary to be more specific about the reasons for my decisions and the scope of this analysis. In the beginning, I mentioned that my purpose of this analysis is to build a "useful" model but here start the first question, what exactly is a "useful" model? this has an answer that may change depending on the user and the problem. On this occasion where the problem is a game, a "useful" model could mean to building a model that plays better than the average human.

Mesure how good is player depends on the game, in this game, there's only one way to obtain points but there are different paths to choose to accomplish that and we need to define the best parameters that copy the behavior we consider appropriate for our standards. To do this we need to test different policies because changing the value of the reward and punishment the model will execute different strategies with the intent of getting the maximum reward.

With all of this in mind, we have to test different combinations of parameters but we don't have infinite time or computational resources, so we need to limit the number of combinations to a number that matches our limitations.

In my case I used google collab (free), this tool has a limit of 12 hours so the decisions that I took to control the number of tests, are:

- 2 types of games (with and without borders)

- 3 different sizes for the input variables where the binary maps change

- Use the same architecture for all the models, using the rule of 2^n for the max number of input variables.

- The parameters of the neural network are constant (learning rate, batch size, optimizer, buffer limit ...)

- 6 different types of policies, where the reward and punish change in value and sign

- Each model have 3 initializations

with these conditions, each model needs approx 10 hours, so in the end, we need 2*3*6*10 = 360 hours (~15 days).




