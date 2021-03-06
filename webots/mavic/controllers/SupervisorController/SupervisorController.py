"""Supervisor controller."""
from UAV import UAV
from PPOAgent import PPOAgent, Transition

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor

env = UAV()
# agent = PPOAgent(numberOfInputs=env.observation_space.shape[0], numberOfActorOutputs=env.action_space.n)
agent = PPOAgent()
solved = False
episodeCount = 0
episodeLimit = 2000
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episodeCount < episodeLimit:
    observation = env.reset()  # Reset robot and get starting observation
    env.episodeScore = 0
    for step in range(env.StepsPerEpisode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selectedAction, actionProb = agent.work(observation, type_="selectAction")
        # Step the supervisor to get the current selectedAction's reward, the new observation and whether we reached
        # the done condition
        newObservation, reward, done, info = env.step([selectedAction])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selectedAction, actionProb, reward, newObservation)
        agent.storeTransition(trans)
        # print(done)
        if done:
            # Save the episode's score
            env.EpisodeScoreList.append(env.EpisodeScore)
            agent.trainStep(batchSize=step + 1)
            solved = env.solved()  # Check whether the task is solved
            break

        env.EpisodeScore += reward  # Accumulate episode reward
        observation = newObservation  # observation for next step is current step's newObservation
    print("Episode #", episodeCount, "score:", env.EpisodeScore)
    episodeCount += 1  # Increment episode counter

if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")
observation = env.reset()
while True:
    selectedAction, actionProb = agent.work(observation, type_="selectActionMax")
    observation, _, _, _ = env.step([selectedAction])
