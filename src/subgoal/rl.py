""" Testing simple reinforcement Q learning on
2 room 7x5 grid world."""

from scipy import *

from pybrain.rl.environments.mazes import Maze, MDPMazeTask
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q
from pybrain.rl.experiments import Experiment

#Create 2d 2room gridworld
structure = array([[1,1,1,1,1,1,1],
                   [1,0,0,1,0,0,1],
                   [1,0,0,1,0,0,1],
                   [1,0,0,0,0,0,1],
                   [1,0,0,1,0,0,1],
                   [1,0,0,1,0,0,1],
                   [1,1,1,1,1,1,1]])

environment = Maze(structure, (1,5))

#Initialize agent doing Q-Learning

controller = ActionValueTable(49, 4)
controller.initialize(0.)

learner = Q()
agent = LearningAgent(controller, learner)

#Create link between agent and environment
task = MDPMazeTask(environment)
experiment = Experiment(task, agent)

while True:
    print "Starting..."
    print environment
    experiment.doInteractions(100)
    agent.learn()
    agent.reset()

    print controller.params.reshape(49,4).max(1).reshape(7,7)
