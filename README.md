Requirements:
import heapq
import pygame
import numpy as np
import heapq
from os import path
from tracemalloc import start
import random
from collections import defaultdict
import matplotlib.pyplot as plt

Description of Project:
In this project we create a snake game using python and create AI agent classes to understand their performance during testing. We include three types of agents using A* search, Safe A* search, and a Reinforcement Learning agent that uses Q tabular learning. In addition to the agents we create a human class that allows users to play the game manually using the UP, DOWN, LEFT, and RIGHT keys. Testing and training methods are used to evaluate the performance of the AI agents.

Testing/Graphs:
In our project we created a test_agent() method that is able to test AI agent performance over a user desired amount of games. 
  method/parameters: test_agent(agent_type, trained_agent=None, games=50)
    -trained_agent parameter is used when using a trained "rl" agent
    -games set to 50 by default unless specified

Since the testing scores are stored in a list, they can be used to create a graphs. We created a graph to compare the A* search and Safe A* search agents scores over the 100 test games. Then used a graph to show the test averages at every 100 episodes for the "rl" agent.

To play the game:
Each agent is able to be played using the play() function with a string containing the type of mode desired: "human", "astar", "safe_astar", or "rl".

*If intenidng to use "rl" agent the agent must must train agent first with number of desired epeisodes*
  example: rl_agent = train_rl(episodes=9000)

In the main.py file, choose between: 
  -"human", "astar", "safe_astar", or "rl" 
    -must change the string value inside the play function to change the game        type from "human", "astar", "safe_astar", or "rl" depending on the desired       ai agent
    
  -for the "rl" agent, must train agent and pass the trained agent                  variable as the "trained_agent" parameter value
        example: play("rl", trained_agent=rl_agent)
