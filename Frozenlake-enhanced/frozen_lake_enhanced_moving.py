import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
import pygame
import random
from os import path
from typing import List, Optional
from gymnasium.utils import seeding

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


# define the maps available for the game (easy to extent S: start, F: Frozen, H: Hole, G: Goal (but changed depending mode))
MAPS = {
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
    "12x12": [
        "SFFFFFFFFFFF",
        "FFFFFFHFFFFF",
        "FFFFFFFFHFFF",
        "FFFFHFFFFFFF",
        "FFFFFFFFFFFF",
        "FFFHFFFFFFFF",
        "FFFFFFFFFFHF",
        "FFFFFFHFFFFF",
        "FFFFFFFFFFFF",
        "FFFFFHFFFFFF",
        "FFFFFFFFFFFF",
        "FFFFFFFFFFFG",
    ],

    
    "16x16": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFHFFFFF",
        "FFFFFHFFFFFFFFFF",
        "FFFFFFFFFFFHFFFF",
        "FFFFFFHFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFHFFFFFFFFFFF",
        "FFFFFFFFFFHFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFHFFFFFFFFF",
        "FFFFFFFFFFFFFHFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFHFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG",
    ],
       "25x25": [
        "SFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFFFFFFFHFFFFFF",
        "FFFFFFFFHFFFFFFFFFFFFFHFF",
        "FFFFFHFFFFFHFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFHHFFFFFFF",
        "FFFFHFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFHFFFFFFFFFFFFFHFF",
        "FFFFFFHFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFHFFFFFFFFFFFF",
        "FFFFHFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFHFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFHFFFFFF",
        "FFFHFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFHFFFFFFFFFFFFFHFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFHFFFFFFFFFFFFFFHF",
        "FFFFFHFFFFFFFFFFFFFHFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFHFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFHFFFFFFFF",
        "FFFFFFFHFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFHFFFF",
        "FFFFHFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFG",
    ],


    "50x50": [
        # 50 characters per line
        "SFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFHFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFGF",
    ]
}

def euclidean_distance_xy(x, y, gx, gy):
    return np.sqrt((x - gx)**2 + (y - gy)**2)


def generate_random_map(
    size: int = 8, p: float = 0.8, seed: Optional[int] = None
) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False
    board = [] 

    np_random, _ = seeding.np_random(seed)

    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]


# Check if the map is valid with a path
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False

#Frozn lake visual interface class
class FrozenLakeEnv(Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, desc=None, map_name="8x8",
                 is_slippery=True, num_agents=1, goal_mode="fixed", keep_dead_alive=False, **kwargs):
        
        self.render_mode = render_mode
        self.num_agents = num_agents
        self.is_slippery = is_slippery
        self.keep_dead_alive = keep_dead_alive
        self.agents_at_goal = [False] * self.num_agents
        if desc is None:
            desc = MAPS[map_name]
        self.desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = self.desc.shape

        nS = self.nrow * self.ncol
        nA = 4

        self.action_space = spaces.Discrete(nA)
        self.observation_space = spaces.Discrete(nS)

        self.episode = 0

        #  Agents 
        self.agent_positions = [(0, 0) for _ in range(self.num_agents)]
        self.episode_steps = 0
        self.agent_rewards = [0.0] * self.num_agents
    
        self.win_counts  = [0] * self.num_agents
        self.hole_counts = [0] * self.num_agents
        self.timeout_counts = [0] * self.num_agents
        self.agent_alive    = [True]  * self.num_agents   # False une fois dans un trou
        self.hole_positions = [None]  * self.num_agents   # case où chaque agent est mort
        self.episode_goals_reached = 0
        self.episode_wins_cumul    = 0
        self.goal_mode = goal_mode

        self.goal_pos = (self.nrow - 1, self.ncol - 1)  # will be set in reset() via goal_mode modes
        

        # --- Pygame ---
        # pygame utils
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")

        self.window_size = (min(64 * self.ncol, 512), min(64 * self.nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.pygame_initialized = False


    # ===========================================================
    # HOLE SENSING : adjacent holes results as a sensor mode
    # ===========================================================
    def sense_adjacent(self, x, y):
        """
        Returns a 4‑element list indicating hole presence in:
        [left, right, up, down] (1 = hole, 0 = safe)
        """
        dirs = [
            (0, -1),  # left
            (0,  1),  # right
            (-1, 0),  # up
            (1,  0)   # down
        ]
        result = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.nrow and 0 <= ny < self.ncol:
                result.append(1 if self.desc[nx][ny] == b"H" else 0)
            else:
                result.append(0)
        return result

    # ===========================================================
    #                   Init a spawn mode to start map
    # ===========================================================


    def _spawn_positions(self):
        """
        Returns num_agents well-spaced starting positions on the grid.
        Each agent is placed on a free cell (no hole, not the goal) at a
        minimum euclidean distance from already-placed agents.
        min_dist scales with grid size so it adapts to all map sizes.
        """
        gx, gy = self.goal_pos
        min_dist = max(2, self.nrow // (self.num_agents + 1))
 
        # All free cells (no hole, not goal)
        free = [
            (r, c)
            for r in range(self.nrow)
            for c in range(self.ncol)
            if self.desc[r][c] != b"H" and (r, c) != (gx, gy)
        ]
 
        if not free:
            # Extreme edge case: no free cells at all
            return [(0, 0)] * self.num_agents
 
        placed = []
        max_attempts = 2000
 
        for _ in range(max_attempts):
            if len(placed) >= self.num_agents:
                break
            idx = self.np_random.integers(0, len(free))
            candidate = free[idx]
 
            # Reject duplicates
            if candidate in placed:
                continue
 
            # Reject cells too close to already-placed agents
            too_close = any(
                euclidean_distance_xy(candidate[0], candidate[1], p[0], p[1]) < min_dist
                for p in placed
            )
            if not too_close:
                placed.append(candidate)
 
        # Fallback: if not enough spaced positions found, relax constraint
        # and fill from remaining free cells not yet picked
        if len(placed) < self.num_agents:
            remaining = [f for f in free if f not in placed]
            # Use numpy RNG correctly — shuffle via integer permutation
            if remaining:
                perm = self.np_random.permutation(len(remaining))
                remaining = [remaining[i] for i in perm]
            placed += remaining[:self.num_agents - len(placed)]
 
        # Last-resort fallback: if still short, repeat last position
        while len(placed) < self.num_agents:
            placed.append(placed[-1] if placed else (0, 0))
 
        return placed[:self.num_agents]


    def reset(self, *,  seed=None, options={"spawn":False}):
        super().reset(seed=seed)


        self.episode_steps = 0
        self.episode_reward = 0.0
        self.episode_goals_reached = 0
        self.episode_wins_cumul    = 0
        self.agents_at_goal = [False] * self.num_agents
        self.agent_alive     = [True] * self.num_agents
        self.hole_positions  = [None] * self.num_agents

        if self.goal_mode == "fixed": # put goal at end of the table [n-1,n-1]
                self.goal_pos = (self.nrow - 1, self.ncol - 1)
        elif self.goal_mode == "random":
            free = [ # find a cell not hole 
                (r, c)
                for r in range(self.nrow)
                for c in range(self.ncol)
                if self.desc[r][c] not in (b"H", b"S")
            ]
            self.goal_pos = free[self.np_random.integers(0, len(free))]

        elif self.goal_mode == "moving":
            self.goal_pos = (self.nrow - 1, self.ncol - 1)  # starts at corner, moves each step (see step())


        if options.get("spawn") : # spwan after goal position to avoid to be at same position
            self.agent_positions = self._spawn_positions()
        else :
            self.agent_positions = [(0, 0) for _ in range(self.num_agents)]

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {"prob": 1}

    def _get_obs(self):
        return tuple(x * self.ncol + y for (x, y) in self.agent_positions)

    # ===========================================================
    #                Movement action and moving goal 
    # ===========================================================

    def apply_action(self, x, y, action):
        if action == LEFT:
            y = max(y - 1, 0)
        elif action == DOWN:
            x = min(x + 1, self.nrow - 1)
        elif action == RIGHT:
            y = min(y + 1, self.ncol - 1)
        elif action == UP:
            x = max(x - 1, 0)
        return x, y

    def move_goal(self):
        moves = [(1,0),(-1,0),(0,1),(0,-1)]
        gx, gy = self.goal_pos
        dx, dy = random.choice(moves)
        nx, ny = gx + dx, gy + dy

        if 0 <= nx < self.nrow and 0 <= ny < self.ncol:
            if self.desc[nx][ny] != b"H":
                self.goal_pos = (nx, ny)

    # ===========================================================
    #                         STEP
    # at each step() return raw reward : goal = +1, hole = -1, step = 0
    # this allows to adapt shaping outside env
    # ===========================================================
    def step(self, actions):

        self.episode_steps += 1

        rewards = [0.0 for _ in range(self.num_agents)]
        terminated = False
        truncated = False


        if self.pygame_initialized:
            # Process user events, key presses
            for event in pygame.event.get():
                if(event.type == pygame.KEYDOWN):
                    # quit game
                    if(event.key == pygame.K_ESCAPE):
                        pygame.quit()
                        import sys;
                        sys.exit()
                    # increase animation speed
                    elif(event.key == pygame.K_EQUALS):
                        self.metadata["render_fps"]+=10
                    # decrease animation speed
                    elif(event.key == pygame.K_MINUS):
                        self.metadata["render_fps"]-=10
                        if(self.metadata["render_fps"]<=0):
                            self.metadata["render_fps"]=1
                    # fastest animation speed. 0 is Unlimited fps
                    elif(event.key == pygame.K_0):
                        self.metadata["render_fps"]=0
                    # reset to original fps
                    elif(event.key == pygame.K_1):
                        self.metadata["render_fps"]=4
                    # toggle rendering. Turn off rendering to speed up training.
                    elif(event.key == pygame.K_9):
                        self.render_mode=None if(self.render_mode=="human") else "human"



        new_positions = list(self.agent_positions) 
        gx, gy = self.goal_pos
        any_goal = False

        # Move every agent
        for i, action in enumerate(actions):
            if not self.agent_alive[i]:
                rewards[i] = 0.0
                continue
 
            if self.agents_at_goal[i]:
                new_positions[i] = (gx, gy)   # keeps the winning agent at goal / due to slippery 
                rewards[i] = 0.0
                continue

            x, y = self.agent_positions[i]
            
            if self.is_slippery:
                # With probability 1/3 take intended action, 1/3 each perpendicular
                perp = {LEFT: [UP, DOWN], RIGHT: [UP, DOWN],
                        UP: [LEFT, RIGHT], DOWN: [LEFT, RIGHT]}
                action = random.choice([action, perp[action][0], perp[action][1]])
            
            new_x, new_y = self.apply_action(x, y, action)
            new_positions[i] = (new_x, new_y)

            # Hole
            if self.desc[new_x][new_y] == b"H":
                rewards[i] = -1
                self.hole_counts[i] += 1
                self.hole_positions[i]  = (new_x, new_y)
                if self.keep_dead_alive:
                    # Mark dead — episode continues for survivors
                    self.agent_alive[i] = False
                else:
                    # Legacy behaviour — end episode immediately
                    terminated = True

            # Catch goal
            if (new_x, new_y) == (gx, gy):
                if not self.agents_at_goal[i]:          # first time reaching goal
                    rewards[i] = 1
                    self.win_counts[i] += 1
                    self.agents_at_goal[i] = True
                    any_goal = True
                else: #normally never here as done if self.agents_at_goal[i]:
                    rewards[i] = 0.0                    # already at goal , do nothing
                    new_positions[i] = (gx, gy) 

        self.agent_positions = new_positions

        # In keep_dead_alive mode: also terminate when ALL agents are dead
        alive_agents   = [i for i in range(self.num_agents) if self.agent_alive[i]]
        all_goal       = all(self.agents_at_goal[i] for i in alive_agents)
        all_dead       = not any(self.agent_alive)

        if any_goal and all_goal:
            terminated = True       # all alive agents reached goal — cooperative win
        elif self.keep_dead_alive and all_dead:
            terminated = True   

        # Move goal
        if not terminated and self.goal_mode == "moving":
            self.move_goal()

        self.episode_reward += sum(rewards)

        return self._get_obs(), rewards, terminated, truncated, {"prob": 1}

    # ===========================================================
    #                   Render interface (based on initial code)
    # ===========================================================

    def render(self):
        if self.render_mode != "human":
            return
        self._render_gui()

    def _render_gui(self):

        if self.window_surface is None:
            pygame.init()
            self.pygame_initialized = True
            self.fixed_cell_size = 100
            pygame.display.init()
            pygame.display.set_caption("Frozen Lake Multi-Agent")

            # Fenêtre normale (pas fullscreen)
            #self.window_surface = pygame.display.set_mode((900, 700))
            self.window_surface = pygame.display.set_mode(flags=pygame.FULLSCREEN)
            self.clock = pygame.time.Clock()

            # Charger les images
            base = path.join(path.dirname(__file__), "img")

            self.ice_img = pygame.transform.scale(
                pygame.image.load(path.join(base, "ice.png")), (self.fixed_cell_size, self.fixed_cell_size)
            )
            self.hole_img = pygame.transform.scale(
                pygame.image.load(path.join(base, "hole.png")), (self.fixed_cell_size, self.fixed_cell_size)
            )
            self.goal_img = pygame.transform.scale(
                pygame.image.load(path.join(base, "goal.png")), (self.fixed_cell_size, self.fixed_cell_size)
            )
            self.agent_img = pygame.transform.scale(
                pygame.image.load(path.join(base, "elf_down.png")), (self.fixed_cell_size, self.fixed_cell_size)
            )

            self.ui_font = pygame.font.SysFont("Courier", 24)
            self.agent_font = pygame.font.SysFont("Courier", 36, bold=True)

        # Clear
        self.window_surface.fill((240, 240, 255))

        # Grid
        for r in range(self.nrow):
            for c in range(self.ncol):
                pos = (c * self.fixed_cell_size, r * self.fixed_cell_size)
                self.window_surface.blit(self.ice_img, pos)
                if self.desc[r][c] == b"H":
                    self.window_surface.blit(self.hole_img, pos)

        # Goal
        gx, gy = self.goal_pos
        pos = (gy * self.fixed_cell_size, gx * self.fixed_cell_size)
        self.window_surface.blit(self.goal_img, pos)


        # Agents — dead agents greyed out
        # Rebuild agent_alive if length drifted out of sync (safety guard)
        if len(getattr(self, 'agent_alive', [])) != len(self.agent_positions):
            self.agent_alive    = [True] * len(self.agent_positions)
            self.hole_positions = [None] * len(self.agent_positions)

        # Normalise scout_idx to a set (supports int, list, or None)
        raw_scout = getattr(self, 'scout_idx', None)
        if raw_scout is None:
            scout_set = set()
        elif isinstance(raw_scout, int):
            scout_set = {raw_scout}
        else:
            scout_set = set(raw_scout)
 
        for i, (ax, ay) in enumerate(self.agent_positions):
            pos      = (ay * self.fixed_cell_size, ax * self.fixed_cell_size)
            alive    = self.agent_alive[i]
            is_scout = i in scout_set
 
            self.window_surface.blit(self.agent_img, pos)
 
            # Grey overlay for dead agents
            if not alive:
                overlay = pygame.Surface(
                    (self.fixed_cell_size, self.fixed_cell_size), pygame.SRCALPHA)
                overlay.fill((180, 180, 180, 160))
                self.window_surface.blit(overlay, pos)
 
            # Badge : circle + number, greyed when dead
            cx = pos[0] + self.fixed_cell_size // 2
            cy = pos[1] + self.fixed_cell_size // 4
            
            if not alive:
                badge_color  = (200, 200, 200)
                border_color = (100, 100, 100)
                text_color   = (120, 120, 120)
            elif is_scout:
                badge_color  = (255, 200,  80)   # orange-yellow for scout
                border_color = (180, 100,   0)
                text_color   = (120,  50,   0)
            else:
                badge_color  = (255, 255, 255)
                border_color = (30,   30,  30)
                text_color   = (20,   20, 180)

            pygame.draw.circle(self.window_surface, badge_color,  (cx, cy), 18)
            pygame.draw.circle(self.window_surface, border_color, (cx, cy), 18, 2)
            label = self.agent_font.render(str(i), True, text_color)
            lw, lh = label.get_size()
            self.window_surface.blit(label, (cx - lw // 2, cy - lh // 2))


        # HUD
        hud_x = self.ncol * self.fixed_cell_size + 20
        line_h   = 28   # pixels par ligne
        hud_y    = 30   # point de départ vertical
        # render actual FPS


        # --- Info block (état courant) ---
        info = [
            f"Episode : {getattr(self, 'episode', 0)}",
            f"Steps   : {self.episode_steps}",
            f"FPS     : {int(self.clock.get_fps())}",
            f"Mode    : {'survive' if self.keep_dead_alive else 'global'}",
            f"Goals   : {getattr(self, 'episode_goals_reached', 0)}/{self.num_agents}",  # ← NEW
            f"Wins    : {getattr(self, 'episode_wins_cumul', 0)}",  
        ]
        
        for i, r in enumerate(getattr(self, 'agent_rewards', [self.episode_reward])):
            alive_str = "alive" if self.agent_alive[i] else "DEAD"
            info.append(f"A{i} [{alive_str}] reward: {r:.2f}")
        

        for i, (ax, ay) in enumerate(self.agent_positions):
            if self.agent_alive[i]:
                dist = euclidean_distance_xy(ax, ay, gx, gy)
                info.append(f"A{i} distance: {dist:.2f}")
            else:
                hx, hy = self.hole_positions[i] if self.hole_positions[i] else (ax, ay)
                info.append(f"A{i} hole at : ({hx},{hy})")
 
        for i, txt in enumerate(info):
            surf = self.ui_font.render(txt, True, (0, 0, 0))
            self.window_surface.blit(surf, (hud_x, hud_y + i * line_h))
 

        # --- Wins / Holes table ---
        table_y = hud_y + len(info) * line_h + 20
        col_w   = 72
        agents  = [f"  A{i}" for i in range(self.num_agents)]
 
        rows = [
            ("Wins",  self.win_counts,  (0,   0, 120)),
            ("Holes", self.hole_counts, (140, 0,   0)),
            ("Timeout", self.timeout_counts, (120, 80,  0)),
        ]
 
        # Header
        header_cols = ["     "] + agents
        for j, h in enumerate(header_cols):
            surf = self.ui_font.render(h, True, (60, 60, 60))
            self.window_surface.blit(surf, (hud_x + j * col_w, table_y))
 
        sep_y = table_y + line_h - 4
        pygame.draw.line(
            self.window_surface, (120, 120, 120),
            (hud_x, sep_y),
            (hud_x + len(header_cols) * col_w, sep_y), 1
        )
 
        for row_idx, (label, counts, color) in enumerate(rows):
            y = table_y + line_h * (row_idx + 1)
            surf = self.ui_font.render(label, True, color)
            self.window_surface.blit(surf, (hud_x, y))
            for j, v in enumerate(counts):
                surf = self.ui_font.render(f"  {v:4d}", True, color)
                self.window_surface.blit(surf, (hud_x + (j + 1) * col_w, y))



        # --- Help block (raccourcis) — commence après info + marge ---
        help_y = table_y + line_h * (len(rows) + 1) + 20
 
        help = [
            f"--- Shortcuts ---",
            f"1 : Reset FPS",
            f"0 : Unlimited FPS",
            f"- : Decrease FPS",
            f"= : Increase FPS",
            f"9 : Toggle render",
            f"ESC : Quit",
        ]
 
        for i, txt in enumerate(help):
            surf = self.ui_font.render(txt, True, (80, 80, 80))
            self.window_surface.blit(surf, (hud_x, help_y + i * line_h))


        # --- Config block — parameters passed via set_config() ---
        config_y = help_y + len(help) * line_h + 20
        cfg = getattr(self, 'config_params', {})
        if cfg:
            surf = self.ui_font.render("--- Config ---", True, (60, 60, 120))
            self.window_surface.blit(surf, (hud_x, config_y))
            config_y += line_h
            for k, v in cfg.items():
                line = f"{k}: {v}"
                surf = self.ui_font.render(line, True, (60, 60, 120))
                self.window_surface.blit(surf, (hud_x, config_y))
                config_y += line_h


        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])


    def set_episode(self, ep):
        self.episode = ep

    def set_steps(self, steps):
        self.episode_steps = steps

    
    def set_agent_rewards(self, rewards):
        self.agent_rewards = list(rewards)

    
    def set_distance(self, dist):
        self.current_distance = dist

    def add_timeout(self):
        for i in range(self.num_agents):
            self.timeout_counts[i] += 1

    def set_scout_idx(self, idx):
        #which indice is a scout
        if idx is None:
            self.scout_idx = []
        elif isinstance(idx, int):
            self.scout_idx = [idx]
        else:
            self.scout_idx = list(idx)
 
    def set_config(self, **kwargs):

        self.config_params = {k: v for k, v in kwargs.items()}

    def set_episode_stats(self, goals_reached, wins_cumul):
        self.episode_goals_reached = goals_reached
        self.episode_wins_cumul    = wins_cumul
