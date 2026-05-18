# unamur-marl4swarmrobotics
Multi-Agent Reinforcement Learning for Swarm Robotics: Evaluating the Impact of Autonomous, Partial, and Full Information Sharing

Code done during master thesis 2025-2026.


Abstract
Imagine a swarm of rovers deployed on the Martian surface, navigating a hazardous terrain
riddled with craters, tasked with locating a distress signal several kilometers away. A
lone rover would probably not survive. Communication with Earth introduces a delay of
up to twenty minutes, making centralized human control impossible in real time. This
master dissertation investigates Multi-Agent Reinforcement Learning (MARL) applied to
swarm robotics, focusing on the impact of information sharing between agents. Four
information sharing architectures are compared: fully decentralized, proximity based, scout
agent, and fully centralized. These configurations are evaluated in a search-and-rescue
inspired environment simulated on a partially observable grid, with fixed, random, and
dynamic objectives. Results show that the communication quality matters more than its
volume. The scout architecture, as a hybrid swarm, offers the best trade-off between
collective performance, agent mortality, and communication cost. Additionally, increasing
the communication radius degrades performance by introducing noise. These findings
introduce the concept of the Amphi-encephalic swarm, defined as a system combining local
autonomy with global collective awareness


Multi-Agent Reinforcement Learning for Swarm
Robotics: Evaluating the Impact of Autonomous,
Partial, and Full Information Sharing
Olivier DUBOIS

Prof : Elio Tuci



Objectives: 
Within a
controlled partially observable grid world inspired by search-and-rescue scenarios, the experimental
work described in this dissertation evaluates four MARL architectures: fully decentralized (none), prox
imity based sharing, scout agent based sharing and fully centralized. Second, it quantifies how each
architecture influences coordination efficiency, task completion rate, agent survival and communication
cost across increasing levels of complexity. Third, it introduces the concept of an Amphi-encephalic
swarm, defined as a system that maintains individual agent autonomy while benefiting from a higher
level information structure. This conceptual term describes and motivates hybrid architectures that go
beyond the classical swarm definition.


Setup : 
> pip install virtualenv 

Then, to create the virtual environment named "venv" type:

> virtualenv venv

or

> python -m venv venv

To activate the virtual environment "venv" type:

> source venv/bin/activate



pip install numpy gymnasium pygame torch

Files from pong folder were used to test Cuda, Google scolar and GPU usage on my laptop.

Test those file before running locally advanced Frozen lake.
(based on : 
 Johnny Code. 2026. Johnny Code — Programming Tutorials and Source Code.
YouTube channel and GitHub repository. https://www.youtube.com/@johnnycode
Source code available at https://github.com/johnnycode8. Accessed: April 2026.)

https://gymnasium.farama.org/


Experiments : 

Files :
frozen_lake_enhanced_moving.py (adapted Frozen lake front end)
frozen_lake_marl_competitive_agent.py (agent reward mechanism)
marl_agent.py (contains agent class)
util_plotter.py (contains utils for graphic/ video purpose)


run_experiment... are used to launch specific config with batch (multiple config in a run, during night;-)
config_....py are used as config files for different modes. DO NOT REMOVE ANY VARIABLE IN IT (as code has no control).

Apex application to load files : 
Create the table:
marl_table_Apex_oracle.sql
Import Apex apps (in 24.1) cloud.oracle.com/free
f110.sql



Remarks: 
AI was used for code review and bug tracking. the launcher scripts (run_xxx were generated to launch our experiments in batch mode. util_ploter was generated to generate video recording and some graphics.

