# main.py
import pygame
from snake_env import SnakeEnv
from human_agent import HumanAgent
from astar_agent import AStarAgent
from rl_agent import QLearningAgent
from safe_astar_agent import SafeAStarAgent

def train_rl(episodes=2000):
    env = SnakeEnv(render_mode=False)   # IMPORTANT: no rendering during training
    agent = QLearningAgent()

    scores = []

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state)  # epsilon-greedy
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        scores.append(env.score)

        if (ep + 1) % 100 == 0:
            avg = sum(scores[-100:]) / 100
            print(f"Episode {ep+1}, average score: {avg:.2f}")

    return agent

def play(agent_type="human", trained_agent=None):
    env = SnakeEnv(render_mode=True)

    if agent_type == "human":
        agent = HumanAgent(env)
    elif agent_type == "astar":
        agent = AStarAgent(env)
    elif agent_type == "rl":
        agent = trained_agent if trained_agent is not None else QLearningAgent()
    elif agent_type == "safe_astar":
        agent = SafeAStarAgent(env)

    else:
        raise ValueError("Unknown agent type")

    state = env.reset()
    done = False

    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, _ = env.step(action)

    # Game over screen
    env.render_game_over()

    # Keep window open until user closes it
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

    print("Game over. Score:", env.score)

def test_agent(agent_type, trained_agent=None, games=50):
    env = SnakeEnv(render_mode=False)

    scored = []
    scores = []

    for _ in range(games):

        if agent_type == "astar":
            agent = AStarAgent(env)
        elif agent_type == "rl":
            agent = trained_agent
        elif agent_type == "safe_astar":
            agent = SafeAStarAgent(env)
        else:
            raise ValueError("Unkown Agent")
        
        state = env.reset()
        done = False
        steps = 0
        max_steps = 1000  # prevent infinite loops

        while not done and steps < max_steps:
            if agent_type == "rl":
                action = agent.act(state, greedy=True)
            else:
                action = agent.act(state)

            state, reward, done, _ = env.step(action)
            steps += 1
        scores.append(env.score)

    avg_score = sum(scores) / len(scores)

    print(f"\n{agent_type.upper()} results over {games} games")
    print(f"Average score: {avg_score:.2f}")
    print(f"Max score: {max(scores)}")
    print(f"Min score: {min(scores)}")

    return scores


if __name__ == "__main__":
    trained_agent = train_rl(episodes=3000)

    rl_scores = test_agent("rl", trained_agent, games=100)
    astar_scores = test_agent("astar", games=100)
    safe_astar_scores = test_agent("safe_astar", games=100)

    play("safe_astar")
