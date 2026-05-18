# all parameters need to be declared in this file (as used in python code as global variable)
# dont delete some even if not used for specific config modes

# These parameters allows to define map and the global game panel
MAP_SIZE = 12           # valid: 4, 8, 12, 16, 25, 50, 100 (only defined in the frozen_lake_enhanced_moving )
NUM_AGENTS_GLOBAL = 4
MAX_STEPS_FACTOR = 12 # map_size*max_step_factor is the maximum number of steps before timeout

EPSILON_DECAY = 0.0002


# Reward management 
# --- global
GOAL_SHAPING = 10 # when goal found 
HOLE_SHAPING = 5 # when hit a hole 
STEP_SHAPING = 0.02 #for each step, to force to find a quick solution without making too much steps
HOLE_STEP_PENALTY = 0.1
BACKTRACK_SHAPING = 0.15 # avoid to go back step same as previous 
DISTANCE_FACTOR = 0.5 # distance from the goal
VISIT_COUNT_DIVIDER = 3 # number of time a cell is visited, to have a mix of exploration and map knowledge.

# --- Competitive
COMPETITIVE_MODE    = False 
COMP_WINNER_BONUS   = 0
COMP_HOLE_PENALTY   = 0
COMP_SURVIVAL_BONUS = 0

# -- cooperative
COOP_MODE = True
COOP_EPISODE_BONUS = 3 #at the end of the episode, we give a bonus for each agent depending the number of agents reaching the gool.
SCOUT_COOP_MULTIPLIER = 2 # we emphase the scout reward, if he helps more, he gets more.


# game parameter
KEEP_DEAD_ALIVE = True # keep player alive when other deads, to allow to finish the game
SPAWN = True # forced initial cell is 0,0, or spawn ramdom
# This parameter must be changed in marl_agent file, but putting here allows to display it in csv file
COMM_RADIUS  = 3 

# Communication mode 
# "none"        : fully decentralised (default)
# "centralized" : every agent sees all others at every step
# "proximity"   : agents share only when within COMM_RADIUS cells
# "scout"       : last agent is the scout, broadcasts to players in range
COMM_MODE = "scout"

# Number of scouts — last NUM_SCOUTS agents become scouts.
# Sample usage  NUM_AGENTS_GLOBAL=4, NUM_SCOUTS=1 means agent 3 is scout
#               NUM_AGENTS_GLOBAL=4, NUM_SCOUTS=2 means agents 2 and 3 are scouts
# Ignored when COMM_MODE != "scout"
NUM_SCOUTS = 1
SCOUT_HOLE_ADJ_PENALTY = 0.5 #penalty if near a hole for agent
SCOUT_GOAL_REWARD_DIVIDER = 1
SCOUT_DIVERGE_FACTOR = 0.1