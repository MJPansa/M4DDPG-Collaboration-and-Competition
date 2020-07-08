import torch as T
from models import DDPGActor, DDPGCritic
from utils import DDPGExperienceBuffer, EnvWrapper, transform_state
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F


class MARL:
    """MARL class that holds multiple agents, critics and buffer, implements step method and training.

        Params
        ======
            env_path(str): path to env
            kwargs(dict): dict holding config values
        """

    def __init__(self, env_path, kwargs):
        self.n_agents = kwargs['n_agents']
        self.env = EnvWrapper(env_path)
        self.agents = [DDPGActor(kwargs) for n in range(self.n_agents)]
        self.critics = [DDPGCritic(kwargs) for n in range(self.n_agents)]
        self.target_agents = [DDPGActor(kwargs) for n in range(self.n_agents)]
        self.target_critics = [DDPGCritic(kwargs) for n in range(self.n_agents)]
        self.buffer = DDPGExperienceBuffer(kwargs['buffer_size'], kwargs['bs'], kwargs['buffer_threshold'],
                                           kwargs['device'])
        self.exp_cache = []

        # holds the stats for each episode
        self.stats = {'rewards': 0.,
                      'actor_loss': 0.,
                      'critic_loss': 0.,
                      'loss': 0.
                      }
        self.rewards = []
        self.steps = 0

        self.gamma = kwargs['gamma']
        self.noise_factor = kwargs['noise_factor']
        self.noise_decay = kwargs['noise_decay']
        self.buffer_size = kwargs['buffer_size']
        self.train_every_n = kwargs['train_every_n']
        self.model_update_every_n = kwargs['model_update_every_n']
        self.clip_grad = kwargs['clip_grad']
        self.tau = kwargs['tau']

    def step(self, states):
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent(states[i]).detach().cpu().squeeze().numpy())
        return actions

    def reset_stats(self):
        self.stats = {key: 0. for key in self.stats}

    def learn(self):

        # train the critics
        for i in range(self.n_agents):
            # sample experience from buffer
            exp_states, exp_actions, exp_rewards, exp_dones, exp_next_states = self.buffer.draw()

            # calculate qvals for state s(t) and s(t+1)
            q_vals = self.critics[i](T.cat([exp_states[0], exp_actions[0]], dim=1),
                                     T.cat([exp_states[1], exp_actions[1]], dim=1))

            next_actions = [self.target_agents[i](exp_next_states[i]) for i in range(self.n_agents)]

            next_q_vals = self.target_critics[i](T.cat([exp_next_states[0], next_actions[0]], dim=1),
                                                 T.cat([exp_next_states[1], next_actions[1]], dim=1))

            next_state_v_one = exp_rewards[i].squeeze() + (
                    self.gamma * next_q_vals.squeeze().detach() * (1 - exp_dones[i].squeeze()))
            critic_loss = F.mse_loss(q_vals.squeeze(), next_state_v_one)

            # minimize critic_loss
            self.critics[i].optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critics[i].parameters(), self.clip_grad)
            self.critics[i].optimizer.step()

            self.stats['critic_loss'] += critic_loss
            self.stats['loss'] += critic_loss

        # train the agents
        for i, agent in enumerate(self.agents):

            exp_states, exp_actions, exp_rewards, exp_dones, exp_next_states = self.buffer.draw()

            exp_actions = [agent(exp_states[n]) for n, agent in enumerate(self.agents)]

            if i == 0:
                actor_loss = -self.critics[i](T.cat([exp_states[0], exp_actions[0]], dim=1),
                                              T.cat([exp_states[1], exp_actions[1].detach()], dim=1)).mean()
            else:
                actor_loss = -self.critics[i](T.cat([exp_states[0], exp_actions[0].detach()], dim=1),
                                              T.cat([exp_states[1], exp_actions[1]], dim=1))[i].mean()

            self.stats['actor_loss'] += actor_loss
            self.stats['loss'] += actor_loss

            agent.optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(agent.parameters(), self.clip_grad)
            agent.optimizer.step()

    def add_exp(self):
        # adds cached experience to buffer
        self.buffer.add(*self.exp_cache)
        self.exp_cache.clear()

    def soft_update(self):
        # soft update
        for n in range(self.n_agents):
            for target_param, local_param in zip(self.target_agents[n].parameters(), self.agents[n].parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

            for target_param, local_param in zip(self.target_critics[n].parameters(), self.critics[n].parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
