import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete

import torch.nn.functional as F


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class MLP(nn.Module):
    def __init__(self,
                 layers,
                 activation=torch.tanh,
                 output_activation=None,
                 output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x


class CategoricalPolicy(nn.Module):
    def __init__(self, in_features, hidden_sizes, activation,
                 output_activation, action_dim):
        super(CategoricalPolicy, self).__init__()

        if hidden_sizes is not None:
            self.logits = MLP(
                layers=[in_features] + list(hidden_sizes) + [action_dim],
                activation=activation)
        else:
            self.logits = MLP(
                layers=[in_features] + [action_dim],
                activation=activation)

    def forward(self, x, a=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None

        return pi, logp, logp_pi


class GaussianPolicy(nn.Module):
    def __init__(self, in_features, hidden_sizes, activation,
                 output_activation, action_dim):
        super(GaussianPolicy, self).__init__()

        self.mu = MLP(
            layers=[in_features] + list(hidden_sizes) + [action_dim],
            activation=activation,
            output_activation=output_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim, dtype=torch.float32))

    def forward(self, x, a=None):
        mu = self.mu(x)
        policy = Normal(mu, self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None

        return pi, logp, logp_pi


class ActorCritic(nn.Module):
    def __init__(self,
                 in_features,
                 action_space,
                 hidden_sizes=(64, 64),
                 activation=torch.tanh,
                 output_activation=None,
                 policy=None):
        super(ActorCritic, self).__init__()

        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(
                in_features,
                hidden_sizes,
                activation,
                output_activation,
                action_dim=action_space.shape[0])
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(
                in_features,
                hidden_sizes,
                activation,
                output_activation,
                action_dim=action_space.n)
        else:
            self.policy = policy(in_features, hidden_sizes, activation,
                                 output_activation, action_space)

        if hidden_sizes is not None:
            self.value_function = MLP(
                layers=[in_features] + list(hidden_sizes) + [1],
                activation=activation,
                output_squeeze=True)
        else:
            self.value_function = MLP(
                layers=[in_features] + [1],
                activation=activation,
                output_squeeze=True)

    def forward(self, x, a=None):
        pi, logp, logp_pi = self.policy(x, a)
        v = self.value_function(x)

        return pi, logp, logp_pi, v

class Actor(nn.Module):
    def __init__(self,
                 in_features,
                 action_space,
                 hidden_sizes=(64, 64),
                 activation=torch.tanh,
                 output_activation=None,
                 policy=None):
        super(Actor, self).__init__()

        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(
                in_features,
                hidden_sizes,
                activation,
                output_activation,
                action_dim=action_space.shape[0])
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(
                in_features,
                hidden_sizes,
                activation,
                output_activation,
                action_dim=action_space.n)
        else:
            self.policy = policy(in_features, hidden_sizes, activation,
                                 output_activation, action_space)

    def forward(self, x, a=None):
        pi, logp, logp_pi = self.policy(x, a)

        return pi, logp, logp_pi

class Critic(nn.Module):
    def __init__(self,
                 in_features,
                 action_space,
                 hidden_sizes=(64, 64),
                 activation=torch.tanh,
                 output_activation=None,
                 policy=None):
        super(Critic, self).__init__()

        if hidden_sizes is not None:
            self.value_function = MLP(
                layers=[in_features] + list(hidden_sizes) + [1],
                activation=activation,
                output_squeeze=True)
        else:
            self.value_function = MLP(
                layers=[in_features] + [1],
                activation=activation,
                output_squeeze=True)

    def forward(self, x, a=None):
        v = self.value_function(x)

        return v



## my ACCNN_model:
class AlexNetCategoricalPolicy(nn.Module):
    def __init__(self, in_features, action_dim, body):
        super(AlexNetCategoricalPolicy, self).__init__()

        self.logits = body(in_features, action_dim)

    def forward(self, x, a=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None

        return pi, logp, logp_pi

class AlexNet(nn.Module):
    def __init__(self, in_channel=4, num_classes=1, hid_input=1024):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channel, 32, kernel_size=(11)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(32, 32, kernel_size=(7), stride=1),
                                      nn.ReLU(),
                                      nn.Dropout2d(0.3, inplace=True),
                                      nn.AvgPool2d(kernel_size=3, stride=1),
                                      nn.Conv2d(32, 64, kernel_size=(5), stride=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.Dropout2d(0.3, inplace=True),
                                      nn.AvgPool2d(kernel_size=3, stride=1),
                                      nn.Conv2d(64, 32, kernel_size=(3)),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(),
                                      nn.Dropout2d(0.3, inplace=True),
                                      nn.Conv2d(32, 16, kernel_size=(3)),
                                      nn.ReLU(),
                                      nn.Dropout2d(0.3, inplace=True),
                                      nn.AvgPool2d(kernel_size=3, stride=2),
                                      )
        self.classifier = nn.Sequential(nn.Dropout(0.3, inplace=True),
                                        nn.Linear(hid_input, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(512, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, num_classes),
                                        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), x[0].numel())
        x = self.classifier(x)
        return x


class ActorAlexNet(nn.Module):

    def __init__(self, in_channel, num_actions):
        super(ActorAlexNet, self).__init__()

        self.policy = AlexNetCategoricalPolicy(in_channel, num_actions, AlexNet)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
            #     nn.init.constant_(module.bias, 0)
            # elif isinstance(module, nn.LSTMCell):
            #     nn.init.constant_(module.bias_ih, 0)
            #     nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, a=None):
        # hx, cx = self.critic_linear(x,)
        pi, logp, logp_pi = self.policy(x, a)
        return pi, logp, logp_pi


class CriticAlexNet(nn.Module):

    def __init__(self, in_channel, num_actions):
        super(CriticAlexNet, self).__init__()
        self.value_function = AlexNet(in_channel, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x, a=None):
        return self.value_function(x)

###############
class AlexLSTMNet(nn.Module):
    def __init__(self, in_channel=4, num_classes=1, hid_input=1024):
        super(AlexLSTMNet, self).__init__()
        self.hid_input = hid_input

        self.features = nn.Sequential(nn.Conv2d(in_channel, 32, kernel_size=(11)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(32, 32, kernel_size=(7), stride=1),
                                      nn.ReLU(),
                                      nn.Dropout2d(0.3),
                                      nn.AvgPool2d(kernel_size=3, stride=1),
                                      nn.Conv2d(32, 64, kernel_size=(5), stride=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.Dropout2d(0.3),
                                      nn.AvgPool2d(kernel_size=3, stride=1),
                                      nn.Conv2d(64, 32, kernel_size=(3)),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(),
                                      nn.Dropout2d(0.3),
                                      nn.Conv2d(32, 16, kernel_size=(3)),
                                      nn.ReLU(),
                                      nn.Dropout2d(0.3),
                                      nn.AvgPool2d(kernel_size=3, stride=2),
                                      )
        self.classifier = nn.Sequential(nn.Dropout(0.3),
                                        nn.Linear(512, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(512, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, num_classes),
                                        )
        self.lstm1 = nn.LSTM(input_size=self.hid_input, hidden_size=512, num_layers=2, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size=512, hidden_size=512,num_layers=1, batch_first=True)

    def forward(self, x):
        x = self.features(x)
        batch_sizes = x.size(0)
        x = x.view(batch_sizes, x[0].numel()).unsqueeze(0)
        if not hasattr(self.lstm1, '_flattened'):
            self.lstm1.flatten_parameters()
            setattr(self.lstm1, '_flattened', True)
        x,  (h_n, h_c) = self.lstm1(x)
        # x, (_, _) = self.lstm2(x, (h_n, h_c))
        x = self.classifier(x.squeeze(0))
        return x

class ActorAlexLSTMNet(nn.Module):

    def __init__(self, in_channel, num_actions):
        super(ActorAlexLSTMNet, self).__init__()

        self.policy = AlexNetCategoricalPolicy(in_channel, num_actions, AlexLSTMNet)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, a=None):
        # hx, cx = self.critic_linear(x,)
        pi, logp, logp_pi = self.policy(x, a)
        return pi, logp, logp_pi

class CriticAlexLSTMNet(nn.Module):

    def __init__(self, in_channel, num_actions):
        super(CriticAlexLSTMNet, self).__init__()
        self.value_function = AlexNet(in_channel, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, a=None):
        return self.value_function(x)

###



if __name__ == "__main__":
    pass
