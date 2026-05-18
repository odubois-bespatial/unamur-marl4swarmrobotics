import gymnasium as gym
import numpy as np
import torch
from marl_agent import Agent
from marl_agent import build_comm_observation
import csv, datetime
from util_plotter import extract_q_maps
from util_plotter import unwrap_env
from util_plotter import save_training_plots

import argparse
import importlib
import sys
import os


# This code is based on https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_q.py
# We adapted starting from this file to extend and provide parameters, extension to multiagent, graphics, and csv stats.




def _apply_coop_episode_bonus(goals_reached, episode_rewards, num_agents, agents):
    # nothing if not coop_mode, or no goal reached
    if not COOP_MODE or COOP_EPISODE_BONUS == 0.0 or goals_reached == 0:
        return
 
    # giving to each agent with a multiplier for scout, 
    for j in range(num_agents):
        multiplier = SCOUT_COOP_MULTIPLIER if agents[j].is_scout else 1.0
        episode_rewards[j] += COOP_EPISODE_BONUS * (goals_reached ** 2) * multiplier # adding more reward if more than one goal reached  (quadratic (au carré))
 
    # Single function for both training and testing.
def run_dqn(episodes=500, render=False, num_agents=2, goal_mode="fixed", training=True,config_name="config",train_episodes=0,model_tag=None):
    
    # Model and CSV tag — encodes all params that affect the model
    # format: {config}_{goal_mode}_ep{train_episodes}
    # example: config_scout_random_ep1000
    _train_ep  = episodes if training else train_episodes
    _model_tag = model_tag or f"{config_name}_{goal_mode}_ep{_train_ep}"

    max_episode_steps = MAP_SIZE*MAX_STEPS_FACTOR
    # loaded here to allow config file externalized (and avoid to reload if already loaded)
    if "FrozenLake-enhanced-v2" not in gym.envs.registry:
        gym.register(
        id="FrozenLake-enhanced-v2",
        entry_point="frozen_lake_enhanced_moving:FrozenLakeEnv",
        kwargs={"map_name": MAP_NAME},
        max_episode_steps=max_episode_steps,
        reward_threshold=0.85,
        )
    
    env = gym.make(
        "FrozenLake-enhanced-v2",
        desc=None, map_name=MAP_NAME,
        is_slippery=True,
        num_agents=num_agents,
        goal_mode=goal_mode,
        keep_dead_alive=KEEP_DEAD_ALIVE,
        render_mode='human' if render else None
    )

    real_env = unwrap_env(env)
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    if COMM_MODE == "scout" and NUM_SCOUTS > 0:
        scout_indices = list(range(num_agents - NUM_SCOUTS, num_agents))
    else:
        scout_indices = []
    scout_idx = scout_indices

    real_env.set_scout_idx(scout_idx)
    real_env.set_config(
        map_name            = MAP_NAME,
        comm_mode           = COMM_MODE,
        agents              = num_agents,
        keep_dead           = KEEP_DEAD_ALIVE,
        spawn               = SPAWN,
        competitive         = COMPETITIVE_MODE,
        coop_mode           = COOP_MODE,
        coop_episode        = COOP_EPISODE_BONUS,
        scout_coop_mult     = SCOUT_COOP_MULTIPLIER,
        goal_shaping        = GOAL_SHAPING,
        hole_shaping        = HOLE_SHAPING,
        dist_factor         = DISTANCE_FACTOR,
        backtrack_shaping   = BACKTRACK_SHAPING,
        step_shaping        = STEP_SHAPING,
        visit_count_divider = VISIT_COUNT_DIVIDER,
        comp_winner_bonus   = COMP_WINNER_BONUS,
        comp_hole_penalty   = COMP_HOLE_PENALTY,
        comp_survival_bonus = COMP_SURVIVAL_BONUS,
        mode                = "TRAINING" if training else "TEST",
        epsilon_decay       = EPSILON_DECAY,
        max_episode_steps    = max_episode_steps
    )

  
    if training: # use parameters
        agents = [
            Agent(
                n_states      = n_states,
                n_actions     = n_actions,
                lr            = 0.001,
                gamma         = 0.99,
                epsilon       = 1.0,
                epsilon_min   = 0.01,
                epsilon_decay = EPSILON_DECAY,
                buffer_size   = 50000,
                beta_start    = 0.4,
                comm_mode     = COMM_MODE,
                comm_radius   = COMM_RADIUS, 
                num_agents    = num_agents,
                is_scout      = (COMM_MODE == "scout" and idx in scout_indices),
                scout_hole_adj_penalty    = SCOUT_HOLE_ADJ_PENALTY if (COMM_MODE == "scout" and idx in scout_indices) else 0.0,
                scout_goal_reward_divider = SCOUT_GOAL_REWARD_DIVIDER if (COMM_MODE == "scout" and idx in scout_indices) else 1,
                scout_diverge_factor      = SCOUT_DIVERGE_FACTOR      if (COMM_MODE == "scout" and idx in scout_indices) else 0.0
            
            )
            for idx in range(num_agents)
        ]
    else: #use saved agents
        agents = []
        for i in range(num_agents):
            is_scout   = (COMM_MODE == "scout" and i in scout_indices)
            ag = Agent(
                n_states   = n_states,
                n_actions  = n_actions,
                comm_mode  = COMM_MODE,
                comm_radius   = COMM_RADIUS, 
                num_agents = num_agents,
                is_scout   = is_scout,
                scout_hole_adj_penalty    = SCOUT_HOLE_ADJ_PENALTY if is_scout else 0.0,
                scout_goal_reward_divider = SCOUT_GOAL_REWARD_DIVIDER if is_scout else 1,
                scout_diverge_factor      = SCOUT_DIVERGE_FACTOR      if is_scout else 0.0,
   
            )
            pth = f"{_model_tag}_agent{i}.pth"
            ag.model.load_state_dict(torch.load(pth))
            print(f"Loaded agent model : {pth}")
            ag.model.eval()
            agents.append(ag)

    inverse_action    = {0: 2, 2: 0, 1: 3, 3: 1}
    all_rewards       = []
    value_map_history = [[] for _ in range(num_agents)]
    save_interval     = 20


  

    # initiate CSV Setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag  = "train" if training else "test"
    log_file  = f"{_model_tag}_{mode_tag}_{timestamp}.csv"

    config_cols  = [
        "mode", "comm_mode", "num_agents", "num_scouts", "scout_indices",
        "competitive_mode", "keep_dead_alive", "goal_mode",
        "map_size", "is_slippery", "comm_radius",
        "coop_mode", "coop_episode_bonus", "scout_coop_multiplier","train_episodes","max_episode_steps"
    ]
    episode_cols = [
        "episode", "steps", "total_reward", "outcome",
        "goal_x", "goal_y", "winner_agent",
        "agents_alive_end", "agents_dead_end",
        "first_death_step", "goals_reached_this_ep", "coop_bonus_applied",
        "holes", "timeouts",
    ]
    agent_cols = []
    for i in range(num_agents):
        agent_cols += [
            f"a{i}_is_scout", f"a{i}_reward", f"a{i}_alive",
            f"a{i}_steps_to_goal", f"a{i}_death_step",
            f"a{i}_end_distance", f"a{i}_min_dist",
            f"a{i}_start_x", f"a{i}_start_y", f"a{i}_start_dist",
            f"a{i}_hole_pos", f"a{i}_cells_visited",
            f"a{i}_backtrack_count", f"a{i}_comm_received",
            f"a{i}_win", f"a{i}_in_hole",
        ]

    fieldnames = config_cols + episode_cols + agent_cols

    with open(log_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(episodes):
            obs   = env.reset(options={"spawn": SPAWN})[0]
            done  = False
            trunc = False
            had_hole_death = False #used to monitor at least one death during the episode

            episode_rewards       = [0.0]  * num_agents
            visit_count           = [np.zeros(n_states) for _ in range(num_agents)]
            prev_actions          = [None] * num_agents
            goal_step             = [-1]   * num_agents
            death_step            = [-1]   * num_agents
            backtrack_count       = [0]    * num_agents
            comm_received         = [0]    * num_agents
            min_dist              = [float('inf')] * num_agents
            goals_reached_this_ep = 0
            first_death_step      = -1
            steps                 = 0
            outcome               = "timeout"

            start_positions = list(real_env.agent_positions)
            gx, gy = real_env.goal_pos 
            real_env.set_episode(ep)

            while not (done or trunc):

                # 1. Build communication messages
                messages = []
                for i in range(num_agents):
                    msg, received = build_comm_observation(
                        agent_idx       = i,
                        obs_scalar      = obs[i],
                        env             = real_env,
                        agent_positions = real_env.agent_positions,
                        agent_alive     = real_env.agent_alive,
                        goal_pos        = real_env.goal_pos,
                        map_size        = MAP_SIZE,
                        comm_mode       = COMM_MODE,
                        comm_radius     =COMM_RADIUS,
                        scout_idx       = scout_idx,
                    )
                    messages.append(msg)
                    if received and real_env.agent_alive[i]: #only counted for living agent
                        comm_received[i] += 1

                # 2. Each agent selects action
                actions = []
                for i in range(num_agents):
                    if not real_env.agent_alive[i]:
                        actions.append(0)
                    else:
                        if not training:
                            agents[i].epsilon = 0.0   # pure exploitation in test
                        actions.append(agents[i].act(obs[i], messages[i]))

                # 3. Environment step
                next_obs, base_rewards, done, trunc, info = env.step(actions)
                steps += 1
                gx, gy = real_env.goal_pos 

                # 4. Build next messages for replay buffer
                next_messages = []
                for i in range(num_agents):
                    nm, _ = build_comm_observation(
                        agent_idx       = i,
                        obs_scalar      = next_obs[i],
                        env             = real_env,
                        agent_positions = real_env.agent_positions,
                        agent_alive     = real_env.agent_alive,
                        goal_pos        = real_env.goal_pos,
                        map_size        = MAP_SIZE,
                        comm_mode       = COMM_MODE,
                        comm_radius     =COMM_RADIUS,
                        scout_idx       = scout_idx,
                    )
                    next_messages.append(nm)

                # 5. Step tracking for CSV
                for i in range(num_agents):
                   
                    ax2, ay2 = real_env.agent_positions[i]
                    d = np.sqrt((ax2 - gx)**2 + (ay2 - gy)**2)
                    if d < min_dist[i]:
                        min_dist[i] = d

                    if not real_env.agent_alive[i] and base_rewards[i] == 0.0:
                        continue
                    if prev_actions[i] is not None:
                        if actions[i] == inverse_action.get(prev_actions[i], -1):
                            backtrack_count[i] += 1

                # 6. Reward shaping
                shaped_rewards = list(base_rewards)
                for i in range(num_agents):
                    s  = obs[i]
                    s2 = next_obs[i]
                    a  = actions[i]
                    r  = base_rewards[i]

                    # Case 1 Dead agent: apply per-step hole penalty
                    # Dead agent — hole step penalty
                    if not real_env.agent_alive[i] and r == 0.0:
                        hole_step_penalty   = -HOLE_STEP_PENALTY
                        shaped_rewards[i]   = hole_step_penalty
                        episode_rewards[i] += hole_step_penalty
                        if training:
                            agents[i].remember(obs[i], a, hole_step_penalty, next_obs[i], done,
                                               message=messages[i], next_message=next_messages[i])
                            agents[i].train_step(batch_size=64)
                        continue
                     # Case 2 Hole collision: amplify negative signal
                    # Hole
                    if r == -1:
                        shaped_rewards[i]   = r * HOLE_SHAPING
                        prev_actions[i]     = a
                        episode_rewards[i] += shaped_rewards[i]
                        # => outcome = "hole" but adapted at the end to manage all kind of configuration. Replaced with had_hole_death to monitor status at the end
                        had_hole_death = True
                        if KEEP_DEAD_ALIVE:
                            real_env.agent_alive[i]    = False
                            new_x, new_y               = divmod(s2, MAP_SIZE)
                            real_env.hole_positions[i] = (new_x, new_y)
                        if death_step[i] == -1:
                            death_step[i] = steps
                            if first_death_step == -1:
                                first_death_step = steps
                        continue

                    # Case 3 Scout agent: dedicated reward function
                    # Scout
                    if agents[i].is_scout:
                        other_scout_pos = [
                            real_env.agent_positions[j]
                            for j in scout_indices
                            if j != i and real_env.agent_alive[j]
                        ]
                        sr = agents[i].scout_reward(
                                s2, MAP_SIZE,
                                real_env.agent_positions,
                                real_env.agent_alive,
                                visit_count[i],
                                STEP_SHAPING,
                                own_adj=messages[i][0:4],
                                other_scout_positions=other_scout_pos if len(scout_indices) > 1 else None
                            )
                        if r == 1:
                            # Reward scales with goals already reached:
                            # promotes arriving last (after players)
                            # adding a shape based on scout agent linked with already achieved goals to promote to be the last arriving.
                            sr += (GOAL_SHAPING / agents[i].scout_goal_reward_divider) * goals_reached_this_ep
                            goals_reached_this_ep += 1   # ← after using the count
                            if goal_step[i] == -1:
                                goal_step[i] = steps
                        else:
                            # If all players are done, switch to distance shaping
                            # to break the 2-scout deadlock
                            # All non-scout agents are done (at goal or dead) force scout to apply distance shaping toward goal.
                            # added to avoid "deadlock betwehn both scout not wanting to close before the other"
                            non_scout_indices = [j for j in range(num_agents) if j not in scout_indices]
                            all_players_done  = all(
                                real_env.agents_at_goal[j] or not real_env.agent_alive[j]
                                for j in non_scout_indices
                            )
                            if all_players_done:
                                old_x, old_y = divmod(s,  MAP_SIZE)
                                new_x, new_y = divmod(s2, MAP_SIZE)
                                d_old = np.sqrt((old_x - gx)**2 + (old_y - gy)**2)
                                d_new = np.sqrt((new_x - gx)**2 + (new_y - gy)**2)
                                sr += DISTANCE_FACTOR * (d_old - d_new)    
                        shaped_rewards[i]   = sr
                        episode_rewards[i] += sr
                        prev_actions[i]     = a
                        continue

                    # Case 4 Goal reached: amplify positive signal
                    # Goal
                    if r == 1:
                        shaped_rewards[i]   = r * GOAL_SHAPING
                        prev_actions[i]     = a
                        episode_rewards[i] += shaped_rewards[i]
                        goals_reached_this_ep += 1
                        # outcome = "goal" => commented to use the logic taking all parameters into account
                        if goal_step[i] == -1:
                            goal_step[i] = steps
                        continue

                    # no new reward for agent already at goal  
                    if real_env.agents_at_goal[i]:
                            shaped_rewards[i] = 0.0
                            continue

                    # Case 5 Normal navigation step
                    # Normal step
                    old_x, old_y = divmod(s,  MAP_SIZE)
                    new_x, new_y = divmod(s2, MAP_SIZE)
                    d_old = np.sqrt((old_x - gx)**2 + (old_y - gy)**2)
                    d_new = np.sqrt((new_x - gx)**2 + (new_y - gy)**2)
                    r += DISTANCE_FACTOR * (d_old - d_new)
                    r -= STEP_SHAPING
                    if prev_actions[i] is not None:
                        if a == inverse_action.get(prev_actions[i], -1):
                            r -= BACKTRACK_SHAPING
                    visit_count[i][s2] += 1
                    r += 1 / np.sqrt(visit_count[i][s2] + 1) / VISIT_COUNT_DIVIDER
                    shaped_rewards[i]   = r
                    prev_actions[i]     = a
                    episode_rewards[i] += shaped_rewards[i]

                # 7. Competitive shaping
                winner_this_step = any(base_rewards[j] == 1 for j in range(num_agents))
                for i in range(num_agents):
                    if agents[i].is_scout:
                        continue
                    if not real_env.agent_alive[i] and base_rewards[i] == 0.0:
                        continue
                    r_base = base_rewards[i]
                    if r_base == 1 and COMPETITIVE_MODE:
                        episode_rewards[i] += COMP_WINNER_BONUS
                        shaped_rewards[i]  += COMP_WINNER_BONUS
                    elif r_base == -1 and COMPETITIVE_MODE:
                        episode_rewards[i] += COMP_HOLE_PENALTY
                        shaped_rewards[i]  += COMP_HOLE_PENALTY
                    elif COMPETITIVE_MODE:
                        episode_rewards[i] += COMP_SURVIVAL_BONUS
                        shaped_rewards[i]  += COMP_SURVIVAL_BONUS

                # 8. Store + train (training only)
                if training:
                    for i in range(num_agents):
                        if not real_env.agent_alive[i] and base_rewards[i] == 0.0:
                            continue
                        if real_env.agents_at_goal[i]:  
                            continue
                        agents[i].remember(
                            obs[i], actions[i], shaped_rewards[i], next_obs[i], done,
                            message=messages[i], next_message=next_messages[i]
                        )
                        agents[i].train_step(batch_size=64)

                obs = next_obs

                if render:
                    real_env.set_steps(steps)
                    real_env.set_agent_rewards(episode_rewards)
                    real_env.set_episode_stats(
                        goals_reached = goals_reached_this_ep,
                        wins_cumul    = sum(real_env.win_counts),
                    )
                    env.render()

            # ── End of episode ────────────────────────────────────────────────

            # Cooperative episode bonus
            coop_bonus_applied = COOP_MODE and COOP_EPISODE_BONUS > 0.0 and goals_reached_this_ep >= 1
            _apply_coop_episode_bonus(goals_reached_this_ep, episode_rewards, num_agents, agents)


            if goals_reached_this_ep > 0 and all(
                real_env.agents_at_goal[i] or not real_env.agent_alive[i] #all agents ended before timeout (dead or alive)
                for i in range(num_agents)
            ):
                # All alive agents reached goal (it could be on the last step (Bug correction)
                outcome = f"goal_{goals_reached_this_ep}of{num_agents}"
                if trunc:
                    real_env.add_timeout()   # still count the timeout for stats
            elif trunc:
                real_env.add_timeout()
                if goals_reached_this_ep == 0:
                    outcome = "timeout"
                else: # timeout with some winds 
                    outcome = f"timeout_{goals_reached_this_ep}win"
            elif goals_reached_this_ep > 0:
                # achieved
                outcome = f"goal_{goals_reached_this_ep}of{num_agents}"

            elif KEEP_DEAD_ALIVE and not any(real_env.agent_alive): #case when continuing after one dead..., but all deads
                outcome = "all_dead"

            elif not KEEP_DEAD_ALIVE and had_hole_death:
                outcome = "hole_death"

            else:
                outcome = "no_goal"

            # Target network update (training only)
            if training and ep % 50 == 0:
                for ag in agents:
                    ag.update_target_network()

            # Value map snapshot (training only)
            if training and ep % save_interval == 0:
                for i, ag in enumerate(agents):
                    V_map, _, _ = extract_q_maps(ag.model, n_states, MAP_SIZE)
                    value_map_history[i].append(V_map)

            all_rewards.append(sum(episode_rewards))

            # CSV row - prepare saving
            gx, gy       = real_env.goal_pos
            winner_agent = min( (i for i in range(num_agents) if goal_step[i] != -1),   key=lambda i: goal_step[i],  default=-1 )
            agents_alive = sum(real_env.agent_alive)

            row = {
                "mode":                  mode_tag,
                "comm_mode":             COMM_MODE,
                "num_agents":            num_agents,
                "num_scouts":            len(scout_indices),
                "scout_indices":         str(scout_indices),
                "competitive_mode":      int(COMPETITIVE_MODE),
                "keep_dead_alive":       int(KEEP_DEAD_ALIVE),
                "goal_mode":             goal_mode,
                "map_size":              MAP_SIZE,
                "is_slippery":           int(real_env.is_slippery),
                "comm_radius":           COMM_RADIUS,
                "coop_mode":             int(COOP_MODE),
                "coop_episode_bonus":    COOP_EPISODE_BONUS,
                "scout_coop_multiplier": SCOUT_COOP_MULTIPLIER,
                "train_episodes" :       train_episodes if not training else episodes,
                "max_episode_steps":     max_episode_steps,
                "episode":               ep,
                "steps":                 steps,
                "total_reward":          round(sum(episode_rewards), 4),
                "outcome":               outcome,
                "goal_x":                gx,
                "goal_y":                gy,
                "winner_agent":          winner_agent,
                "agents_alive_end":      agents_alive,
                "agents_dead_end":       num_agents - agents_alive,
                "first_death_step":      first_death_step,
                "goals_reached_this_ep": goals_reached_this_ep,
                "coop_bonus_applied":    int(coop_bonus_applied),
                "holes":           sum(1 for i in range(num_agents) if death_step[i] != -1),
                "timeouts":        int(trunc),
            }
            for i in range(num_agents):
                ax, ay   = real_env.agent_positions[i]
                alive    = real_env.agent_alive[i]
                hole_pos = real_env.hole_positions[i]
                sx, sy   = start_positions[i]
                start_d  = np.sqrt((sx - gx)**2 + (sy - gy)**2)
                end_d    = np.sqrt((ax - gx)**2 + (ay - gy)**2)
                md       = min_dist[i] if min_dist[i] < float('inf') else end_d
                row[f"a{i}_is_scout"]        = int(i in scout_indices)
                row[f"a{i}_reward"]          = round(episode_rewards[i], 4)
                row[f"a{i}_alive"]           = int(alive)
                row[f"a{i}_steps_to_goal"]   = goal_step[i]
                row[f"a{i}_death_step"]      = death_step[i]
                row[f"a{i}_end_distance"]        = round(end_d, 4)
                row[f"a{i}_min_dist"]        = round(md, 4)
                row[f"a{i}_start_x"]         = sx
                row[f"a{i}_start_y"]         = sy
                row[f"a{i}_start_dist"]      = round(start_d, 4)
                row[f"a{i}_hole_pos"]        = f"{hole_pos[0]},{hole_pos[1]}" if hole_pos else ""
                row[f"a{i}_cells_visited"]   = int(np.sum(visit_count[i] > 0))
                row[f"a{i}_backtrack_count"] = backtrack_count[i]
                row[f"a{i}_comm_received"]   = comm_received[i]
                row[f"a{i}_win"]      = int(goal_step[i] != -1)
                row[f"a{i}_in_hole"]     = int(death_step[i] != -1)
            writer.writerow(row)

            print(f"[{'TRAIN' if training else 'TEST'}] Episode {ep} | " f"outcome={outcome} | steps={steps} | "f"reward={sum(episode_rewards):.3f} | "f"alive={agents_alive}/{num_agents} | "f"goals={goals_reached_this_ep}")
            for i, r in enumerate(episode_rewards):
                print(f"  Agent {i} ({'scout' if agents[i].is_scout else 'player'}) = {r:.3f}")

    print(f"\nResults saved to {log_file}")
    env.close()

    if training: #save training plts
        for i, ag in enumerate(agents):
            pth = f"{_model_tag}_agent{i}.pth"
            torch.save(ag.model.state_dict(), pth)
            print(f"Saved agent model : {pth}")
        save_training_plots(agents, all_rewards, value_map_history, env, n_states, MAP_SIZE,tag=_model_tag)
    return all_rewards



#  MAIN

if __name__ == "__main__":

    # Load --config argument from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config_proximity",           # default is config.py if not specified
        help="Load config module with parameters e.g. --config config_proximity"
    )

    parser.add_argument(
        "--mode",
        choices=["train", "run", "both"],
        default="both",
        help="train = training only | run = run only | both = train then test"
    )

    parser.add_argument(
        "--train_episodes",
        default=150,
        type=int,
        help="number of episodes used for training"
    )

    parser.add_argument(
        "--test_episodes",
        default=200,
        type=int,
        help="number of episodes used for test"
    )

    parser.add_argument(
        "--render_training",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="render training with visual interface (slower)"
    )

    parser.add_argument(
        "--render_run",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="render run with visual interface (slower)"
    )

    parser.add_argument(
        "--goal_mode",
        choices=["fixed", "random", "moving"],
        default="fixed",
        help="choose the goal target mode : fixed - Goal stays at (MAP_SIZE-1, MAP_SIZE-1) every episode, every step   / random - Goal picked randomly at episode start, fixed during episode / moving - Goal takes a random walk one cell per step  "
    )

    parser.add_argument("--test_goal_mode",
    choices=["fixed", "random", "moving"],
    default=None,   # None = same as train goal_mode
    help="goal mode for test phase — overrides goal_mode for run only"
    )

    parser.add_argument(
        "--model_tag",
        default=None,
        help="Explicit model tag to load for test/cross-test. "
             "Example: config_scout_random_ep1000. "
             "If omitted, built automatically from config+goal_mode+episodes."
    )


    args = parser.parse_args()

    # Inject all variables into global scope (no control here, for dev purpose only)
    cfg = importlib.import_module(args.config)
    globals().update({k: v for k, v in vars(cfg).items() if not k.startswith("_")})
    globals()["MAP_NAME"] = f"{MAP_SIZE}x{MAP_SIZE}" # define map name from map_size

    print(f"Running with config: {args.config}")
    print(f"  MAP_SIZE={MAP_SIZE} | COMM_MODE={COMM_MODE} | NUM_AGENTS={NUM_AGENTS_GLOBAL}")

    test_goal = args.test_goal_mode or args.goal_mode #to avoid issue if test_goal_mode is not specified

    if args.mode in ("train", "both"):
        run_dqn(episodes=args.train_episodes, render=args.render_training, num_agents=NUM_AGENTS_GLOBAL, goal_mode=args.goal_mode, training=True,config_name=args.config,train_episodes=args.train_episodes,model_tag=args.model_tag)
        print("Training completed.")
    # Test model:
    if args.mode in ("run", "both"):
        run_dqn(episodes=args.test_episodes, render=args.render_run,  num_agents=NUM_AGENTS_GLOBAL, goal_mode=test_goal, training=False,config_name=args.config,train_episodes=args.train_episodes,model_tag=args.model_tag)
