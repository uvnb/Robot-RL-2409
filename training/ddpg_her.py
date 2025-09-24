import numpy as np
import gymnasium as gym
import panda_gym
from agents.ddpg import DDPGAgent
from utils.HER import her_augmentation
import matplotlib.pyplot as plt


if __name__ == "__main__":

    n_games = 30
    opt_steps = 64
    best_score = 0
    score_history = []
    avg_score_history = []
    actor_loss_history = []
    critic_loss_history = []
    success_history = []
    distance_to_goal_history = []


    num_success = 0
    threshold = 0.05  # hoặc giá trị phù hợp với môi trường của bạn (2.5cm)




    env = gym.make('PandaReach-v3')
    obs_shape = env.observation_space['observation'].shape[0] + \
                env.observation_space['achieved_goal'].shape[0] + \
                env.observation_space['desired_goal'].shape[0]

    agent = DDPGAgent(env=env, input_dims=obs_shape)

    actor_loss_episode = None
    critic_loss_episode = None
    for i in range(n_games):
        done = False
        truncated = False
        score = 0
        step = 0

        obs_array = []
        actions_array = []
        new_obs_array = []

        observation, info = env.reset()

        while not (done or truncated):
            curr_obs, curr_achgoal, curr_desgoal = observation.values()
            state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal))

            # Choose an action
            action = agent.choose_action(state, False)

            # Excute the choosen action in the environement
            new_observation, _, done, truncated, _ = env.step(np.array(action))
            next_obs, next_achgoal, next_desgoal = new_observation.values()
            new_state = np.concatenate((next_obs, next_achgoal, next_desgoal))

            # Compute reward
            reward = 1.0 - np.linalg.norm(next_achgoal - next_desgoal)

            # Store experience in the replay buffer
            agent.remember(state, action, reward, new_state, done)
        
            obs_array.append(observation)
            actions_array.append(action)
            new_obs_array.append(new_observation)

            observation = new_observation
            score += reward
            step += 1
        



        # Lấy achieved_goal và desired_goal cuối cùng của episode
        achieved_goal = observation['achieved_goal']
        desired_goal = observation['desired_goal']

        if np.linalg.norm(achieved_goal - desired_goal) < threshold:
            num_success += 1




        success = int(np.linalg.norm(achieved_goal - desired_goal) < threshold)
        success_history.append(success)


        #khoảng cách tới đích
        distance_to_goal = np.linalg.norm(achieved_goal - desired_goal)
        distance_to_goal_history.append(distance_to_goal)




        # Augmente replay buffer with HER
        her_augmentation(agent, obs_array, actions_array, new_obs_array)

        # train the agent in multiple optimization steps
        for _ in range(opt_steps):
            actor_loss, critic_loss = agent.learn()
            if actor_loss is not None:
                actor_loss_episode = actor_loss
            if critic_loss is not None:
                critic_loss_episode = critic_loss

        if actor_loss_episode is not None and critic_loss_episode is not None:
            print(f" Actor loss = {actor_loss_episode:.4f}, Critic loss = {critic_loss_episode:.4f}")
        else:
            print(f" Actor loss = N/A, Critic loss = N/A")
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)
        actor_loss_history.append(actor_loss_episode)
        critic_loss_history.append(critic_loss_episode)

        if avg_score > best_score:
            best_score = avg_score
        
        print(f"Episode {i} steps {step} score {score:.1f} avg score {avg_score:.1f}")




    success_rate = num_success / n_games
    print(f"Success rate: {success_rate*100:.2f}%")




    window = 20  # hoặc 10, 50 tuỳ bạn
    success_rate_avg = np.convolve(success_history, np.ones(window)/window, mode='valid')




    np.savez('results.npz', score_history=score_history, avg_score_history=avg_score_history)

    plt.figure(figsize=(10,5))
    plt.plot(score_history, label='Score per Episode')
    plt.plot(avg_score_history, label='Average Score (last 100)')
    plt.plot(actor_loss_history, label='Actor Loss per Episode')
    plt.plot(critic_loss_history, label='Critic Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Training Progress & Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(success_history, label='Success (1=success, 0=fail)', alpha=0.3)
    plt.plot(np.arange(window-1, len(success_history)), success_rate_avg, label=f'Success Rate (avg {window})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Success')
    plt.title('Success Rate per Episode')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('success_rate.png')
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(distance_to_goal_history, label='Distance to Goal per Episode')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.xlabel('Episode')
    plt.ylabel('Distance to Goal')
    plt.title('Distance to Goal per Episode')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('distance_to_goal.png')
    plt.close()
