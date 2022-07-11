from tensorforce.environments import Environment
from dataclasses import dataclass
from tensorforce.agents import Agent

# L'environnement donne l'etat à partir d'une action


class PhicompEnvironment(Environment):

    def __init__(self, get_state_from_action_function):
        super().__init__()
        self.number_of_actions = 8 # 8 actions possible

        self.shape_states = (8,) # 8 states possibles (alerts)

        self.get_state_from_action = get_state_from_action_function  # get events, clean, predict_alerts

    # process : get_states, calculate actions, execute_action and get_next_states

    # Returns the state space specification
    def states(self):
        return dict(type='float', shape=self.shape_states)

    # Returns the action space specification.
    def actions(self):
        return dict(type='int', num_values=self.number_of_actions)

    # Executes the given action(s) and advances the environment by one step.
    def execute(self, action):

        next_state = self.get_state_from_action(action)

        return next_state



class AgentCreator:


    def __init__(self):
        self.agent = 'ppo', 'tensorforce'

    def create(self, environment):
        return Agent.create(agent=self.agent, environment=environment, update=64,
            optimizer=dict(optimizer='adam', learning_rate=1e-3),
            objective='policy_gradient', reward_estimation=dict(horizon=20)
        )


from tensorforce import Runner
class PhicompRunner:

    def __init__(self, dict_param_environment, dict_param_agent,max_episode_timesteps=500):

        self.environment = PhicompEnvironment(**dict_param_environment)

        self.agent = AgentCreator(**dict_param_agent).create(self.environment)

        self.max_episode_timesteps = max_episode_timesteps


    def run(self, number_of_episode=200):

        # Initialize the runner
        runner = Runner(agent=self.agent, environment=self.environment,
                        max_episode_timesteps=self.max_episode_timesteps)

        # Train for 200 episodes
        runner.run(num_episodes=number_of_episode)

        # close it
        runner.close()

@dataclass
class ReinforcementLearning:

    def __post_init__(self):
        pass

    def states(self):
        pass

    def actions(self):
        pass

    def execute(self, actions):
        next_state, end_program, reward = ',', '', ''
        return next_state, end_program, reward


def main():
    # Create environment
    environment = Environment.create(environment='environment.json', max_episode_timesteps=500)

    # Create agent
    agent = Agent.create(agent='agent.json', environment=environment)

    # Train for 100 episodes
    for _ in range(100):
        terminal = False

        # reset environment
        states = environment.reset()

        while not terminal:
            # get action from states
            actions = agent.act(states=states)

            # get states from actions
            states, terminal, reward = environment.execute(actions=actions)

             # calibrate agent
            agent.observe(terminal=terminal, reward=reward)