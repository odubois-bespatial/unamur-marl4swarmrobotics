import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors


# ---------------------------------------------------------
#  One-hot encoding for discrete states
# ---------------------------------------------------------
def one_hot(state, n_states):
    v = torch.zeros(n_states)
    v[state] = 1.0
    return v


# ---------------------------------------------------------
#  Utility: unwrap env for GUI overlay
# ---------------------------------------------------------
def unwrap_env(env):
    real_env = env
    while hasattr(real_env, "env"):
        real_env = real_env.env
    return real_env

def extract_q_maps(model, n_states, map_size):
    V = np.zeros(n_states)
    P = np.zeros(n_states)
    U = np.zeros(n_states)

    model.eval()

    # Detect the model's actual input size (may be larger than n_states
    # when communication messages are concatenated to the one-hot state).
    try:
        n_input = model.net[0].in_features   # DQN or PaddedModelWrapper wrapping DQN
    except AttributeError:
        n_input = n_states                   # fallback — plain model
    pad_size = max(0, n_input - n_states)

    for s in range(n_states):
        oh = one_hot(s, n_states)
        if pad_size > 0:
            oh = torch.cat([oh, torch.zeros(pad_size)])
        inp = oh.unsqueeze(0)
        with torch.no_grad():
            qvals = model(inp).cpu().numpy()[0]

        V[s] = np.max(qvals)              # Value map
        P[s] = np.argmax(qvals)           # Policy map
        U[s] = np.var(qvals)              # Uncertainty map

    return (
        V.reshape(map_size, map_size),
        P.reshape(map_size, map_size),
        U.reshape(map_size, map_size)
    )



def create_value_map_animation(value_maps, map_size, filename="value_map_animation.mp4"):
    """
    Crée une animation montrant l'évolution de la Value Map pendant l'entraînement.
    value_maps : liste de matrices map_size x map_size
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_idx):
        ax.clear()
        ax.set_title(f"Value Map – step {frame_idx}")
        im = ax.imshow(value_maps[frame_idx], cmap="viridis", vmin=np.min(value_maps[0]), vmax=np.max(value_maps[-1]))
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(value_maps),
        interval=150,
        blit=True
    )

    ani.save(filename, writer="ffmpeg", dpi=150)
    plt.close()


def plot_raw_map(raw_map, map_size,tag=""):
    """
    Affiche la carte brute FrozenLake avec couleurs :
    S = vert, G = rouge, H = bleu, F = gris clair.
    """

    # Matrice numérique pour afficher les couleurs
    num_map = np.zeros((map_size, map_size))

    # Encodage :
    # F = 0 (sol)
    # H = 1 (hole)
    # S = 2 (start)
    # G = 3 (goal)
    for i in range(map_size):
        for j in range(map_size):
            c = raw_map[i, j]
            if c == "F":
                num_map[i, j] = 0
            elif c == "H":
                num_map[i, j] = 1
            elif c == "S":
                num_map[i, j] = 2
            elif c == "G":
                num_map[i, j] = 3

    # Définition des couleurs
    cmap = colors.ListedColormap([
        "#DDDDDD",  # 0 : F (sol normal, gris clair)
        "#4A90E2",  # 1 : H (hole, bleu)
        "#00CC44",  # 2 : S (start, vert)
        "#FF3333"   # 3 : G (goal, rouge)
    ])

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6, 6))
    plt.imshow(num_map, cmap=cmap, norm=norm)
    plt.xticks([]); plt.yticks([])
    plt.title("FrozenLake raw map")
    plt.tight_layout()
    fname = f"{tag}_raw_map.png" if tag else "raw_map.png"
    plt.savefig(fname)
    plt.close()


def extract_raw_map(env):
    """
    Récupère la carte brute du FrozenLake (F, H, S, G)
    depuis l'environnement unwrapé.
    """
    real_env = unwrap_env(env)
    raw = real_env.desc  # tableau bytes
    # Conversion en tableau de strings
    return np.array([[c.decode('utf-8') for c in row] for row in raw])


def plot_q_maps(V_map, P_map, U_map, map_size,i=0,tag=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Value Map
    ax = axes[0]
    im = ax.imshow(V_map, cmap='viridis')
    ax.set_title("Value Map (V)")
    fig.colorbar(im, ax=ax)

    # Policy Map
    ax = axes[1]
    im = ax.imshow(P_map, cmap='tab20')
    ax.set_title("Policy Map (argmax Q)")
    fig.colorbar(im, ax=ax)

    # Uncertainty Map
    ax = axes[2]
    im = ax.imshow(U_map, cmap='magma')
    ax.set_title("Uncertainty Map (Var(Q))")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    fname = f"{tag}_q_maps_agent{i}.png" if tag else f"q_maps{i}.png"
    plt.savefig(fname)
    plt.close()



def plot_policy_arrows(P_map, map_size,i=0,tag=""):
    """
    Affiche la Policy Map sous forme de flèches directionnelles.
    P_map : matrice map_size x map_size contenant pour chaque case l'action optimale.
    """

    # Actions FrozenLake :
    # 0 = Left, 1 = Down, 2 = Right, 3 = Up
    action_to_arrow = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.zeros((map_size, map_size)), cmap="Greys", alpha=0.0)
    ax.set_title("Policy Map (flèches)")

    for i in range(map_size):
        for j in range(map_size):
            action = int(P_map[i, j])
            arrow = action_to_arrow[action]
            ax.text(j, i, arrow, ha='center', va='center', fontsize=16)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fname = f"{tag}_policy_arrows_agent{i}.png" if tag else f"policy_arrows{i}.png"
    plt.savefig(fname)
    plt.close()

# Sauvegarde des différents fichiers des graphiques
def save_training_plots(agents, all_rewards, value_map_history, env, n_states, map_size,tag=""):

    # 1. Per-agent Q-maps and policy arrows
    for i, ag in enumerate(agents):
        V_map, P_map, U_map = extract_q_maps(ag.model, n_states, map_size)
        plot_q_maps(V_map, P_map, U_map, map_size, i,tag=tag)
        plot_policy_arrows(P_map, map_size, i,tag=tag)
        _pre = f"{tag}_" if tag else ""
        print(f"Agent {i}: Q-maps saved to {_pre}q_maps_agent{i}.png, {_pre}policy_arrows_agent{i}.png")

    # 2. Raw map
    raw_map = extract_raw_map(env)
    plot_raw_map(raw_map, map_size,tag=tag)
    print(f'Raw map saved to {"" + tag + "_" if tag else ""}raw_map.png')

    # 3. Smoothed reward curve
    window   = min(20, len(all_rewards))
    smoothed = np.convolve(all_rewards, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(all_rewards, alpha=0.3, label="raw")
    plt.plot(smoothed, label=f"{window}-ep average")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Curve")
    plt.legend()
    plt.tight_layout()
    fname = f"{tag}_reward_curve.png" if tag else "reward_curve_dqn.png"
   
    plt.savefig(fname)
    plt.close()
    print(f'Reward curve saved to {"" + tag + "_" if tag else ""}reward_curve.png')

    # 4. Value map animations
    for i in range(len(agents)):
        if len(value_map_history[i]) > 1:
            _anim = f"{tag}_value_map_agent{i}.mp4" if tag else f"value_map_agent_{i}.mp4"
            create_value_map_animation(
                value_map_history[i],
                map_size,
                filename=_anim
            )
            print(f"Agent {i}: value map animation saved to {_anim}")