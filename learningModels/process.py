import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 
import time

#Tf agents 
#Environment
from tf_agents.environments import utils
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments import tf_py_environment

#Agents
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

#Policies
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
#Replay buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
#Trajectories
from tf_agents.trajectories import trajectory

#Game 
from learningModels.GameEnv import SnakeGameEnv

class LearningProcess:
    """All process for reinforce learning"""

    #Instance of the game environment
    env = SnakeGameEnv()
    #Limits in each episodes
    limit_env_train = TimeLimit(env, 1000)
    limit_env_eval = TimeLimit(env, 1000)
    #Tensor environment
    train_env = tf_py_environment.TFPyEnvironment(limit_env_train)
    eval_env = tf_py_environment.TFPyEnvironment(limit_env_eval)

    #HYPERPARAMETERS
    num_iterations = 200
    #Replay parameters
    initial_collect_steps = 1000
    collect_steps_per_iteration = 1
    replay_buffer_max_length = 100000
    #Learning parameters
    learning_rate = 0.00025
    batch_size = 500
    #Performance analysis parameters
    log_interval = 50
    num_eval_episodes = 10
    eval_interval = 100

    
    def network(self):
        """Define the deep network"""

        fc_layer_param = (128,128,128)

        self.q_agent = q_network.QNetwork(
            self.train_env.observation_spec(),  #Define input (State)
            self.train_env.action_spec(),       #Define Output (actions)
            fc_layer_params=fc_layer_param      #Hidden layers
        )

    
    def def_agent(self):
        """Define the agent (architecture of the solver)"""

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate)
        train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(), 
            self.train_env.action_spec(),
            q_network = self.q_agent,
            optimizer = optimizer,
            td_errors_loss_fn = common.element_wise_squared_loss,
            train_step_counter = train_step_counter
        )

        self.agent.initialize()

    
    def policy(self):
        """Policies for the initial data collection"""
        self.random_policy = random_tf_policy.RandomTFPolicy(
                                                    self.train_env.time_step_spec(),
                                                    self.train_env.action_spec()
                                                    )

    def def_replay_buffer(self):
        """Save a determinated number of steps"""
        #Define the parameters 
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec = self.agent.collect_data_spec,
            batch_size = self.train_env.batch_size,
            max_length = self.replay_buffer_max_length
        )
    

    def collect_data(self):
        """Collect the initial data for the buffer"""
        for _ in range(self.initial_collect_steps):
            collect_step(self.train_env, self.random_policy, self.replay_buffer)
        
        #Define the lots of data to be passed
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls = 3,
            sample_batch_size = self.batch_size,
            num_steps = 2
        ).prefetch(3)
        
        #Make it the dataset iterable
        self.iterator = iter(dataset)


    def pre_learning_process(self):
        """Simplifies the necesary steps before running the network"""
        self.network()
        self.def_agent()
        self.policy()
        self.def_replay_buffer()
        self.collect_data()

    def training(self):
        """Training process"""

        self.agent.train = common.function(self.agent.train)
        #Counter of steps
        self.agent.train_step_counter.assign(0)
        #Compute the average return (network without training)
        avg_return = compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
        returns = [avg_return]
        losses = []


        for _ in range(self.num_iterations):

            for _ in range(self.collect_steps_per_iteration):
                #Save the step make it with the actual neural network
                collect_step(self.train_env, self.agent.collect_policy, self.replay_buffer)
                #Previous states for replay experience
                experience, unused_info = next(self.iterator)
                #Loss of the network 
                train_loss = self.agent.train(experience).loss
                #Actual step
                step = self.agent.train_step_counter.numpy()

                if step % self.log_interval == 0:
                    print('step = {}: loss = {}'.format(step, train_loss))
                    losses.append(train_loss)

                if step % self.eval_interval == 0:
                    #Evaluates the behaviour while the network is training
                    avg_return = compute_avg_return(self.eval_env, self.agent.policy, self.num_eval_episodes)
                    print('step = {}: Average Return = {}'.format(step, avg_return))
                    returns.append(avg_return)
                    
                
        self.visualization(returns, losses)
        self.policy_saver()

        return returns, losses
    

    def visualization(self, returns, losses):
        """"""
        fig, ax = plt.subplots(1, 2)
        #Title
        fig.suptitle('Training process performance', fontsize=16)
        #Data
        iterations_avg = range(0, self.num_iterations + 1, self.eval_interval)
        iterations_loss = range(0, self.num_iterations, self.log_interval)
        #Graph 1
        ax[0].plot(iterations_avg, returns)
        ax[0].set_ylabel('Average Return')
        ax[0].set_xlabel('Iterations')
        ax[0].set_title('Evolution of the Average Return ')
        #Graph 2
        ax[1].plot(iterations_loss, losses)
        ax[1].set_ylabel('Loss')
        ax[1].set_xlabel('Iterations')
        ax[1].set_title('Evolution of the loss')
        plt.show()


    def policy_saver (self):
        
        save_dir = self._save_points_dir()
        tf_policy_saver = policy_saver.PolicySaver(self.agent.policy)
        tf_policy_saver.save(save_dir)
        print('\nThe policy of the deep Q network is saved in: "{}"'.format(save_dir))
  

    def get_save_policy(self, screen):
        saved_policy = tf.compat.v2.saved_model.load(self._save_points_dir())
        num_episodes = 1
        time_step = self.eval_env.reset()
        
        items = {
			'mobile_items':
							{'Snake': self.env.snake},
			'static_items':
							{'Prey': self.env.snake.prey}
		}
        
        while not time_step.is_last():
            action_step = saved_policy.action(time_step) 
            time_step = self.eval_env.step(action_step.action)
            #print(self.env.snake.complete_mapping())
            screen.start_pygame()
            screen.update_window_frame(items)
            time.sleep(0.2)
            

    def _save_points_dir(self):
        current_path = os.path.abspath(os.getcwd())
        policy_dir = os.path.join(current_path, 'control_point')
        return policy_dir






def collect_step(environment, policy, buffer):
    """Execute the step in the environment and add it to the buffer following the respective policy"""

    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)


def compute_avg_return(environment, policy, num_episodes=10):
    """Execute the network predictions into the enivronment"""
    total_return = 0.0

    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        
        total_return += episode_return
    
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]








        

    








