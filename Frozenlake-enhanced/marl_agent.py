import torch
import torch.optim as optim
import random
import torch.nn as nn
import numpy as np


# ==========================================================================
#  COMMUNICATION MODES
#  Set on each Agent via agent comm_mode 
#    "none" : no sharing (default, fully decentralised)
#    "centralized" : full sharing: every agent sees all others data
#    "proximity"  : local sharing: agent shares only with neighbours within
#                    COMM_RADIUS euclidean distance
#    "scout"      : one dedicated scout agent broadcasts to all players in
#                    its radius; scout has its own reward function
# ==========================================================================
 
# 8 floats per neighbor:
# [pos_x, pos_y, dist_goal, alive, adj_left, adj_right, adj_up, adj_down]
MSG_SIZE = 8




#  build_comm_observation : generate communication message
 
def build_comm_observation(
    agent_idx,
    obs_scalar,
    env,
    agent_positions,
    agent_alive,
    goal_pos,
    map_size,
    comm_mode,
    comm_radius,  
    scout_idx=None,
):
    """
    Returns (enriched_obs_tensor, msg_received: bool).
 
    Builds the augmented observation for agent[agent_idx] based on the chosen communication mode. 
    If no communication applies, returns a zero-padded tensor of consistent size.
     Encoding per neighbour slot: [pos_x/map, pos_y/map, dist/diag, alive] — normalised to [0,1] so the network gets stable inputs.
    """
    n = len(agent_positions)
    diag = np.sqrt(2) * map_size          # max possible distance
    
    # Normalise scout_idx to a set for O(1) tests (to initiate same size for each mode)
    if scout_idx is None:
        scout_set = set()
    elif isinstance(scout_idx, int):
        scout_set = {scout_idx}
    else:
        scout_set = set(scout_idx)
    
    # Message allocation. one slot per possible other agent (n-1 slots total) + 4 to put own hole adjacency)
    message = np.zeros(4+(n - 1) * MSG_SIZE, dtype=np.float32)
    received = False
 
    def _fill_slot(j):
        """Encode agent j into a normalised slot."""
        jx, jy = agent_positions[j]
        gx, gy = goal_pos
        dist_j = np.sqrt((jx - gx)**2 + (jy - gy)**2)
        adj = env.sense_adjacent(jx, jy) if env is not None else [0,0,0,0]

        return np.array([
            jx / map_size,
            jy / map_size,
            dist_j / diag,
            float(agent_alive[j]),
            adj[0], adj[1], adj[2], adj[3],
        ], dtype=np.float32)
 
    ax, ay = agent_positions[agent_idx]


    # Agent SELF-ADJACENCY:
    # message[0..3] = [hole_left, hole_right, hole_up, hole_down] around self
    # sense will fill in as a sensor.
    own_adj = env.sense_adjacent(ax, ay) if env is not None else [0, 0, 0, 0]
    message[0:4] = np.array(own_adj, dtype=np.float32)
 
    # - CENTRALIZED: every agent sees every other agent
    if comm_mode == "centralized":
        slot_idx = 0
        for j in range(n):
            if j == agent_idx:
                continue
            if agent_alive[j]: #not sending position if agent is dead
                offset = 4 + slot_idx * MSG_SIZE
                message[offset:offset + MSG_SIZE] = _fill_slot(j)
                received = True
            slot_idx += 1
 
    # - PROXIMITY: see only agents within COMM_RADIUS
    elif comm_mode == "proximity":
        slot_idx = 0
        for j in range(n):
            if j == agent_idx:
                continue
            jx, jy = agent_positions[j]
            dist_ij = np.sqrt((ax - jx)**2 + (ay - jy)**2)
            if dist_ij <= comm_radius  and agent_alive[j]:
                offset = 4 + slot_idx * MSG_SIZE
                message[offset:offset + MSG_SIZE] = _fill_slot(j)
                received = True
            slot_idx += 1
 
     # - SCOUT: player receives from any scout in range 
    elif comm_mode == "scout" and scout_set:
        if agent_idx not in scout_set:
            # Player: merge broadcast from every alive scout within range ( manage multi scout messages to take the first)
            got_broadcast = False
            for s_idx in scout_set:
                if not agent_alive[s_idx]:
                    continue
                sx, sy = agent_positions[s_idx]
                dist_to_scout = np.sqrt((ax - sx)**2 + (ay - sy)**2)
                if dist_to_scout <= comm_radius :
                    # This scout is in range — fill all other-agent slots
                    slot_idx = 0
                    for j in range(n):
                        if j == agent_idx:
                            continue
                        offset = 4 + slot_idx * MSG_SIZE
                        message[offset:offset + MSG_SIZE] = _fill_slot(j)
                        slot_idx += 1
                    got_broadcast = True
                    break   # one scout broadcast is enough take the first in range
            received = got_broadcast
        else :
            slot_idx = 0
            for j in range(n):
                if j == agent_idx:
                    continue
                if j in scout_set and agent_alive[j]:
                    # Fill this agent's slot with the other scout's state
                    offset = 4 + slot_idx * MSG_SIZE
                    message[offset:offset + MSG_SIZE] = _fill_slot(j)
                    received = True
                slot_idx += 1
        # Scouts themselves get no external message (never obs stays zero-padded
 
    if comm_mode == "none":
        return message[0:4], received

    return message, received
 
 
# =========================================================================
#  AGENT class
# =========================================================================




class Agent:
    def __init__(
        self,
        n_states,
        n_actions,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.0005,
        buffer_size=50000,
        beta_start=0.4,
        # Communication mode
        comm_mode="none",       # "none" | "centralized" | "proximity" | "scout"
        num_agents=1,           # total number of agents. This is needed for msg sizing (allocation)
        is_scout=False,         # True if this agent is the scout
        scout_coverage_factor=0.3,
        scout_proximity_bonus=0.5,
        scout_hole_adj_penalty=0.3,
        scout_goal_reward_divider=2,
        scout_diverge_factor=0.1,
        comm_radius=3,
    ):
        # Hyperparamètres
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.beta = beta_start

        self.comm_mode      = comm_mode
        self.comm_radius = comm_radius
        self.num_agents     = num_agents
        self.is_scout       = is_scout
        self.scout_coverage_factor = scout_coverage_factor
        self.scout_proximity_bonus = scout_proximity_bonus
        self.scout_hole_adj_penalty = scout_hole_adj_penalty
        self.scout_goal_reward_divider = scout_goal_reward_divider
        self.scout_diverge_factor      = scout_diverge_factor
 
        # Input size = agent adjacent holes + n_states (one-hot) + message slots for other agents
        msg_extra    = 4 + ((num_agents - 1) * MSG_SIZE if comm_mode != "none" else 0)
        n_input      = n_states + msg_extra
 
        # Dimensions
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_input   = n_input

        # Réseaux DDQN
        self.model = DQN(n_input, n_actions)
        self.target_model = DQN(n_input, n_actions)
        self.target_model.load_state_dict(self.model.state_dict())

        # Optimiseur
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Mémoire PER
        self.memory = PrioritizedReplayBuffer(capacity=buffer_size)

   

    # --------------------------------------------------------------- 
    #  Build full input tensor: one-hot state + communication message
    # ----------------------------------------------------------------
    def build_input(self, state_scalar, message=None):
        """
        Concatenates the one-hot state with the communication message vector.
        If comm_mode is "none" or message is None, input is just the one-hot state.
        """
        oh = one_hot(state_scalar, self.n_states)
        if self.n_input == self.n_states:
            # comm_mode == "none" no message slot in network
            return oh
        # With communication: pad zeros if message is None or wrong size
        msg_size = self.n_input - self.n_states
        if message is not None and len(message) == msg_size:
            msg_t = torch.tensor(message, dtype=torch.float32)
        else:
            msg_t = torch.zeros(msg_size, dtype=torch.float32)
        return torch.cat([oh, msg_t])
 


    # -------------------------------------------------------
    #  ACTIONS
    # -------------------------------------------------------
    def act(self, state,message=None):
        """Retourne une action  (ε-greedy)."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        state_tensor = self.build_input(state, message).unsqueeze(0)
        with torch.no_grad():
            qvals = self.model(state_tensor)
            return torch.argmax(qvals).item()


    # ------------------------------------------------------------------
    #  Scout reward shaping (called from training loop at each step)
    # ------------------------------------------------------------------
    def scout_reward(self, state_scalar, map_size, agent_positions, agent_alive,
                     visit_count, step_penalty,own_adj=None,other_scout_positions=None):
        """
        Compute the scout reward.
        Replaces the standard shaping for this agent when is_scout=True.
        """
        x, y = divmod(state_scalar, map_size)
 
        # Coverage bonus
        visit_count[state_scalar] += 1
        r = self.scout_coverage_factor / np.sqrt(visit_count[state_scalar] + 1)
 
        # Proximity bonus reward being close to at least one alive player
        for j, (px, py) in enumerate(agent_positions):
            if not agent_alive[j]:
                continue      # skip dead agents
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist > 0 and dist <= self.comm_radius:   # dist>0 excludes self agent
                r += self.scout_proximity_bonus
                break         # one nearby player is enough (takes the first)
 
        if own_adj is not None:
            holes_adjacent = sum(own_adj)          # 0 to 4
            if holes_adjacent > 0:
                r -= self.scout_hole_adj_penalty * holes_adjacent          # -0.5 per adjacent hole

        #Divergence bonus to ncourages scouts to spread across the map.
        if other_scout_positions:
            max_dist = np.sqrt(2) * map_size
            for sx, sy in other_scout_positions:
                dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                r += self.scout_diverge_factor * (dist / max_dist)

        r -= step_penalty
        return r

    # -------------------------------------------------------
    #  STOCKAGE
    # -------------------------------------------------------
    def remember(self, state, action, reward, next_state, done,message=None, next_message=None):
        #The state stored includes the communication message so the network trains on the full enriched input.
        self.memory.push(state, action, reward, next_state, done,message, next_message)

    # -------------------------------------------------------
    #  ENTRAINEMENT (PER + DDQN)
    # -------------------------------------------------------
    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        # Sample a prioritised batch from the replay buffer
        sample = self.memory.sample(batch_size, beta=self.beta,
                            msg_size=self.n_input - self.n_states)
        if sample is None:
            return

        (states, actions, rewards, next_states, dones, messages, next_messages, indices, weights) = sample

        # Build enriched input tensors (one-hot state+comm message)
        states_t  = torch.vstack([
            self.build_input(s, m)
            for s, m in zip(states, messages)
        ])

        next_st   = torch.vstack([
            self.build_input(s, m)
            for s, m in zip(next_states, next_messages)
        ])
        
        actions_t = torch.tensor(actions)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        dones_t   = torch.tensor(dones,   dtype=torch.float32)
        
        # Current Q-values from the online network
        qvals          = self.model(states_t)
        qvals_selected = qvals[range(batch_size), actions_t]


        # ---- Double DQN ----
        with torch.no_grad():
            # Online network selects the action
            online_actions = self.model(next_st).argmax(1)
            # ... target network evaluates it (avoids overestimation)
            next_qvals     = self.target_model(next_st)[range(batch_size), online_actions]
            targets        = rewards_t + self.gamma * next_qvals * (1 - dones_t)

        # TD-error and PER-weighted loss
        # TD-error
        td_errors = targets - qvals_selected
        # Loss PER
        loss      = (weights * (td_errors ** 2)).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update replay buffer priorities with new TD-errors
        self.memory.update_priorities(indices, td_errors.abs().detach())

        # Decay epsilon (exploration -> exploitation)
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    # -------------------------------------------------------
    #  UPDATE DU TARGET NETWORK
    # -------------------------------------------------------
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # ---------------------------------------------------------
#  DQN Neural Network
# ---------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
#  One-hot encoding for discrete states
# ---------------------------------------------------------
def one_hot(state, n_states):
    v = torch.zeros(n_states)
    v[state] = 1.0
    return v



class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = []
        self.pos = 0

    def push(self, s, a, r, s2, done, message=None, next_message=None):
        max_priority = max(self.priorities, default=1.0)
        entry = (s, a, r, s2, done, message, next_message)

        if len(self.memory) < self.capacity:
            self.memory.append(entry)
            self.priorities.append(max_priority)
        else:
            self.memory[self.pos] = entry
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4, msg_size=0):
        if len(self.memory) == 0:
            return None

        # Probabilités basées sur les priorités^alpha
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)

        samples = [self.memory[i] for i in indices]

        # Poids d’importance
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()

        s, a, r, s2, d, msgs, next_msgs = zip(*samples)

        # Normalise None messages to zero arrays
        def _norm(m_list):
            return [
                m if m is not None else np.zeros(msg_size, dtype=np.float32)
                for m in m_list
            ]
 
        msgs      = _norm(list(msgs))
        next_msgs = _norm(list(next_msgs))
 
        return (s, a, r, s2, d, msgs, next_msgs,
                indices, torch.tensor(weights, dtype=torch.float32))

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td.item()) + 1e-5

    def __len__(self):
        return len(self.memory)