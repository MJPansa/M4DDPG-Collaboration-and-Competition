import numpy as np
import torch as T
from agent import MARL

PATH = 'Tennis_Linux/Tennis.x86_64'

args = {'obs_space': 24,  # length of state vector
        'action_space': 2,  # number of actions
        'n_agents': 2,  # number of agents
        'n_hidden_actor': (128, 256, 128),  # number of hidden neurons per layer
        'n_hidden_critic': (512, 256),
        'bs': 128,  # number of samples per batch
        'lr_actor': 1e-4,  # learning rate for actor network
        'lr_critic': 1e-3,  # learning rate for critic network
        'device': 'cuda:0',  # device to use for computations
        'gamma': .99,  # discount factor
        'noise_factor': 0.5,  # noise factor for action noise
        'noise_decay': 0.999,  # decay of noise factor applied after every update
        'noise_minimum': 0.03,  # minimum of noise factor
        'buffer_size': 100_000,  # buffer size/length of experience buffer
        'episodes': 10000,  # maximum of episodes to train for
        'buffer_threshold': .10,  # buffer threshold before starting training
        'train_every_n': 1,  # train on one batch every n steps of the env
        'model_update_every_n': 2,  # soft update of target models every n steps
        'train_n_epochs': 2, # train n times/epochs
        'tau': 0.01,  # value for soft updating the target networks tau*local+(1-tau)*target
        'clip_grad': 1.,  # clipping value for gradients
        'exploration_steps': 10_000  # number of random explorations steps before using network predictions
        }

maddpg = MARL(PATH, kwargs=args)

for episode in range(args['episodes']):

    done = False
    states = maddpg.env.reset()
    while not done:
        maddpg.exp_cache.append([T.Tensor(state) for state in states])

        # begin with explorations steps and sample from uniform distribution
        if maddpg.steps < args['exploration_steps']:
            action_one = np.random.uniform(-1, 1, size=(args['action_space']))
            action_two = np.random.uniform(-1, 1, size=(args['action_space']))
            actions = [action_one, action_two]
        else:
            # get actions from MARL
            action_one, action_two = maddpg.step(states)

            # add noise to network actions
            noise_one = np.random.randn(args['action_space']) * max(args['noise_factor'], args['noise_minimum'])
            noise_two = np.random.randn(args['action_space']) * max(args['noise_factor'], args['noise_minimum'])
            actions = [np.clip(action_one + noise_one, -1., 1.), np.clip(action_two + noise_two, -1., 1.)]

        maddpg.exp_cache.append([T.Tensor(action) for action in actions])

        # take step in the environment
        next_states, rewards, done_flags, _ = maddpg.env.step(actions)

        maddpg.stats['rewards'] += (max(rewards))

        maddpg.steps += 1

        # store transitions
        maddpg.exp_cache.append([T.Tensor([reward]) for reward in rewards])
        maddpg.exp_cache.append([T.Tensor([flag]) for flag in done_flags])
        maddpg.exp_cache.append([T.Tensor(next_state) for next_state in next_states])

        maddpg.add_exp()

        if np.any(done_flags):
            done = True

        states = next_states.copy()

        # only train the network every n steps AND when threshold is reached
        if (maddpg.steps % maddpg.train_every_n == 0) and maddpg.buffer.threshold:

            for n in range(maddpg.train_every_n):
                maddpg.learn()

            # update the noise factor with decay
            if maddpg.steps % 3 == 0:
                args['noise_factor'] *= args['noise_decay']

        # do a soft update of the target networks every n steps with value tau
        if maddpg.steps % maddpg.model_update_every_n == 0:
            maddpg.soft_update()

    # append the episode rewards
    maddpg.rewards.append(maddpg.stats['rewards'])

    print(f'episode {episode}:')
    print(f'rewards: {maddpg.stats["rewards"]:.2f}')
    print(f'actor loss: {maddpg.stats["actor_loss"]:.3f}')
    print(f'critic loss: {maddpg.stats["critic_loss"]:.3f}')
    print(f'noise factor: {max(args["noise_factor"], args["noise_minimum"]):.4f}')
    print(f'buffer size: {len(maddpg.buffer)}\n')

    maddpg.reset_stats()
    # # save models every 100 episodes
    if episode % 500 == 0:
        for n in range(maddpg.n_agents):
            T.save(maddpg.agents[n].state_dict(), f'actor_{n}_eps_{episode}_rew_{maddpg.stats["rewards"]:.3f}.h5')
            T.save(maddpg.critics[n].state_dict(), f'critic_{n}_eps_{episode}_rew_{maddpg.stats["rewards"]:.2f}.h5')

    # env is considered solved after mean rewards of +30, save models
    if np.mean(np.array(maddpg.rewards[-100:])) > 0.5:
        print(f'SOLVED ENV AFTER {episode} EPISODES')
        for n in range(maddpg.n_agents):
            T.save(maddpg.agents[n].state_dict(), f'solved_actor_{n}_eps_{episode}_rew_{maddpg.stats["rewards"]:.2f}.h5')
            T.save(maddpg.critics[n].state_dict(), f'solved_critic_{n}_one_eps_{episode}_rew_{maddpg.stats["rewards"]:.2f}.h5')

        break
