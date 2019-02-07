[image2]: score.png "Score"

### Learning Algorithm

This project uses the DDPG training algorithm which is considered an actor-critic algorithm. It was first introduced by [this paper](https://arxiv.org/abs/1509.02971). This method is a development over the Deep Q Learning method so that it can be used in a continuous action domain. The paper states that it is impossible to apply Q Learning to continuous action spaces because finding the greedy policy requires an optimization of a<sub>t</sub> at every time step and this optimization is too slow thus rendering this method unusable. Instead, they follow an actor-critic method
based on DPG.

The DDPG algorithm maintains an actor function which specifies the current policy deterministically mapping states to specific actions. The critic Q(s,a) is learned using the Bellman's equation. Since using Neural Networks as nonlinear function approximators makes the convergence guarantees dissapear, techniques such as Minibatch training and Replay Buffers are used. A major challenge in learning in continuous spaces is exploration. Because DDPG is an offline policy learning algorithm, we can treat the exploration separately and we do that by adding noise sampled from an Ornstein-Uhlenbeck noise process to the actor policies. Check the paper for more details.

The two tennis players are trained using a DDPG like algorithm. Each player has its own actor but they share the critic and the experience buffer. This algorithm is inspired from [the paper](https://arxiv.org/pdf/1706.02275.pdf) on MADDPG . This algorithm comes to help in multi-agent environments where traditional RL approaches like Q-Learning and policy gradient don't fare very well.

The model for the actor is :
```python
self.seed = torch.manual_seed(seed)
self.fc1 = nn.Linear(state_size, fc1_units)
self.fc2 = nn.Linear(fc1_units, fc2_units)
self.fc3 = nn.Linear(fc2_units, action_size)
self.bn = nn.BatchNorm1d(fc1_units)
```

with the forward function :
```python
x = F.relu(self.fc1(state))
x = self.bn(x)
x = F.relu(self.fc2(x))
return F.tanh(self.fc3(x))
```

The model for the critic is :
```python
self.seed = torch.manual_seed(seed)
self.fcs1 = nn.Linear(state_size*num_agents + action_size*num_agents, fcs1_units)
self.fc2 = nn.Linear(fcs1_units, fc2_units)
self.fc3 = nn.Linear(fc2_units, 1)
self.bn1 = nn.BatchNorm1d(fcs1_units)
```

with the forward function :
```python
x = torch.cat((states, actions), dim=1)
x = F.relu(self.fcs1(x))
x = self.bn1(x)
x = F.relu(self.fc2(x))
return self.fc3(x)
```
Check out [model.py](model.py) for the full code.
Both models use batch normalization to make the learning easier. The first hidden layer has 128 nodes and the second hidden layer has 64 nodes.The learning rate I chose for the actor and the critic is 1e-4. This rate was chosen by trial and error.

The file [SharedCritic.py](SharedCritic.py) contains an actor for every agent and a common critic that is used to teach both actors. Every state transition is saved into the experience buffer that has a capacity of 100000 transitions. This experience buffer is shared by the actors. The model uses a batch size of 256.

The system is considered solved when we maintain an average score of .50 over the last 100 episodes. My version solves the problem in 3152 steps. The following graph represents the score of the agent over episodes :

![Score][image2]

If you want to see how the trained agent performs, load [Tennis-SharedCritic-TrainedModel.ipynb](Tennis-SharedCritic-TrainedModel.ipynb)

The algorithm could be further improved by ,for example, changing the sampling of the steps used for learning. Currenly we sample uniformly from the Replay Buffer, but we could do prioritized experince replay, which means that we give higher weight more meaningful transitions, thus sampling them more frequently.
