import torch as T
import torch.optim as optim
import torch.nn.modules as nn
import torch.nn.functional as F


class DDPGActor(nn.Module):
    """Actor network for DDPG, outputs actions given a state.

        Params
        ======
            n_states(int): number of inputs, ie. vector length of state
            n_actions(int): number of outputs, ie vector length of actions
            n_hidden(int): number of hidden neurons per layer
            lr(float): learning rate to be passed to the optimizer
            device(string): device to use for computations, 'cuda' or 'cpu'
        """

    def __init__(self, kwargs):
        super(DDPGActor, self).__init__()
        self.n_states = kwargs['obs_space']
        self.n_actions = kwargs['action_space']
        self.n_hidden = kwargs['n_hidden_actor']
        self.lr = kwargs['lr_actor']
        self.device = kwargs['device']

        self.input = nn.Linear(self.n_states, self.n_hidden[0])
        self.l1 = nn.Linear(self.n_hidden[0], self.n_hidden[1])
        self.l2 = nn.Linear(self.n_hidden[1], self.n_hidden[2])
        self.out = nn.Linear(self.n_hidden[2], self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)

    def forward(self, x):
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x).unsqueeze(0).to(self.device)

        x = F.relu(self.input(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.tanh(self.out(x))

        return x


class DDPGCritic(nn.Module):
    """Critic network for DDPG, outputs state-action value given a state-action input.

        Params
        ======
            n_states(int): number of inputs, ie. vector length of state
            n_actions(int): number of outputs, ie vector length of actions
            n_hidden(int): number of hidden neurons per layer
            lr(float): learning rate to be passed to the optimizer
            device(string): device to use for computations, 'cuda' or 'cpu'
        """

    def __init__(self, kwargs):
        super(DDPGCritic, self).__init__()
        self.n_states = kwargs['obs_space']
        self.n_actions = kwargs['action_space']
        self.n_hidden = kwargs['n_hidden_critic']
        self.n_agents = kwargs['n_agents']
        self.lr = kwargs['lr_critic']
        self.device = kwargs['device']

        self.input = nn.Linear((self.n_states + self.n_actions) * self.n_agents, self.n_hidden[0])
        self.l1 = nn.Linear(self.n_hidden[0], self.n_hidden[1])
        self.out = nn.Linear(self.n_hidden[1], 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)

    def forward(self, x, y):
        x = F.relu(self.input(T.cat([x, y], dim=1)))
        x = F.relu(self.l1(x))
        x = self.out(x)

        return x
