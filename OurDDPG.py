import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
                super(Actor, self).__init__()

                self.l1 = nn.Linear(state_dim, 400)
                self.l2 = nn.Linear(400, 300)
                self.l3 = nn.Linear(300, action_dim)
                
                self.max_action = max_action

        
        def forward(self, x):
                x = F.relu(self.l1(x))
                x = F.relu(self.l2(x))
                x = self.max_action * torch.tanh(self.l3(x)) 
                return x 


class Critic(nn.Module):
        def __init__(self, state_dim, action_dim):
                super(Critic, self).__init__()

                self.l1 = nn.Linear(state_dim + action_dim, 400)
                self.l2 = nn.Linear(400, 300)
                self.l3 = nn.Linear(300, 1)


        def forward(self, x, u):
                x = F.relu(self.l1(torch.cat([x, u], 1)))
                x = F.relu(self.l2(x))
                x = self.l3(x)
                return x

class BetaNN(nn.Module):
        def __init__(self, state_dim, action_dim, condition_on_action=False):
                super(BetaNN, self).__init__()

                self.condition_on_action = condition_on_action
                input_dim = state_dim + action_dim if condition_on_action else state_dim
                self.l1 = nn.Linear(input_dim, 400)
                self.l2 = nn.Linear(400, 300)
                self.l3 = nn.Linear(300, 1)

        def forward(self, x, u):
            input = torch.cat([x, u], dim=1) if self.condition_on_action else x
            x = F.relu(self.l1(input))
            x = F.relu(self.l2(x))
            x = F.sigmoid(self.l3(x))
            return x


class DDPG(object):
        def __init__(self, state_dim, action_dim, max_action, args):
                self.actor = Actor(state_dim, action_dim, max_action).to(device)
                self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
                self.actor_target.load_state_dict(self.actor.state_dict())
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

                self.critic = Critic(state_dim, action_dim).to(device)
                self.critic_target = Critic(state_dim, action_dim).to(device)
                self.critic_target.load_state_dict(self.critic.state_dict())
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters())              

                if args.n_backprop > 1: 
                    self.beta = BetaNN(state_dim, action_dim, args.action_conditional_beta).to(device)
                    self.beta_target = BetaNN(state_dim, action_dim, args.action_conditional_beta).to(device)
                    self.beta_target.load_state_dict(self.beta.state_dict())
                    self.beta_optimizer = torch.optim.Adam(self.beta.parameters(), lr=args.beta_lr)


        def select_action(self, state):
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                return self.actor(state).cpu().data.numpy().flatten()

        def query_beta(self, state, action):
            state  = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)

            if len(state.size()) == 2: 
                state, action = state.unsqueeze(0), action.unsqueeze(0)

            return self.beta_target(state, action).cpu().data.numpy()


        def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, n_backprop=1):
                for it in range(iterations):

                        # Sample replay buffer 
                        x, y, u, r, d = replay_buffer.sample(batch_size, n_backprop)
                        state         = torch.FloatTensor(x).to(device)
                        action        = torch.FloatTensor(u).to(device)
                        next_state    = torch.FloatTensor(y).to(device)
                        done          = torch.FloatTensor(1 - d).to(device) # actually not_done
                        reward        = torch.FloatTensor(r).to(device)
                        
                        n_backprop = state.size(1)

                        for t in range(n_backprop):
                            state_t      = state[:, t]
                            action_t     = action[:, t]
                            next_state_t = next_state[:, t]
                            done_t       = done[:, t]
                            reward_t     = reward[:, t]

                            # Get current Q estimate
                            current_Q_t = self.critic(state_t, action_t)
                            
                            # Compute the target Q value
                            target_Q_t = self.critic_target(next_state_t, self.actor_target(next_state_t))
                            target_Q_t = reward_t + (discount * done_t * target_Q_t).detach()

                            if n_backprop >  1:

                                # set up the recurrence
                                if t == 0: 
                                    prev_value = current_Q_t.detach()

                                # get beta value
                                beta_t = self.beta(state_t, action_t)
                                prev_value = (1-done_t) * current_Q_t + (done_t) * prev_value
                                prev_value = beta_t * current_Q_t + (1 - beta_t) * prev_value - reward_t
                            else:
                                # Good ol' DDPG --> next block of code will -re- add the last reward_t, 
                                # so let's cancel that to write as little code as possible
                                prev_value = current_Q_t - reward_t

                        # compute temporally mixed Q estimate
                        current_Q = prev_value + reward_t
                        
                        # pick the last Q value as target (i.e. t + n for n step return @ timestep t)
                        target_Q  = target_Q_t

                        # Compute critic loss
                        critic_loss = F.mse_loss(current_Q, target_Q)

                        # Optimize Beta Network

                        if n_backprop > 1:
                            self.beta_optimizer.zero_grad()
                            critic_loss.backward(retain_graph=True)
                            self.beta_optimizer.step()

                        # Optimize the critic
                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        self.critic_optimizer.step()
                        

                        # Compute actor loss
                        actor_loss = -self.critic(state[:, 0], self.actor(state[:, 0])).mean()
                        
                        # Optimize the actor 
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        # Update the frozen target models
                        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                        if n_backprop > 1:
                            for param, target_param in zip(self.beta.parameters(), self.beta_target.parameters()):
                                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        def save(self, filename, directory):
                torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
                torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


        def load(self, filename, directory):
                self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
                self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
