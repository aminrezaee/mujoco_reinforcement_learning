from .agent import Agent
from entities.features import Run
from models.actor import Actor
from models.critic import Critic
import torch
class PPOAgent(Agent):
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=Run.instance().training_config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=Run.instance().training_config.learning_rate)
        
    def act(self, state:torch.Tensor) -> torch.Tensor:
        return self.actor(state)
    
    def train(self):
        pass
    
    def save(self):
        experiment_path = Run.instance().experiment_path
        current_episode = Run.instance().dynamic_config.current_episode
        torch.save(self.actor.state_dict(), f"{experiment_path}/{current_episode}/actor.pth")
        torch.save(self.critic.state_dict(), f"{experiment_path}/{current_episode}/critic.pth")
        torch.save(self.actor_optimizer.state_dict(), f"{experiment_path}/{current_episode}/actor_optimizer.pth")
        torch.save(self.critic_optimizer.state_dict(), f"{experiment_path}/{current_episode}/critic_optimizer.pth")
        
    
    def load(self):
        experiment_path = Run.instance().experiment_path
        current_episode = Run.instance().dynamic_config.current_episode
        self.actor_state_dict = torch.load(f"{experiment_path}/{current_episode}/actor.pth")
        self.actor = Actor()
        self.actor.load_state_dict(self.actor_state_dict)
        
        self.critic_state_dict = torch.load(f"{experiment_path}/{current_episode}/critic.pth")
        self.critic = Critic()
        self.critic.load_state_dict(self.critic_state_dict)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=Run.instance().training_config.learning_rate)
        self.actor_optimizer_state_dict = torch.load(f"{experiment_path}/{current_episode}/actor_optimizer.pth")
        self.actor_optimizer.load_state_dict(self.actor_optimizer_state_dict)
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=Run.instance().training_config.learning_rate)
        self.critic_optimizer_state_dict = torch.load(f"{experiment_path}/{current_episode}/critic_optimizer.pth") 
        self.critic_optimizer.load_state_dict(self.critic_optimizer_state_dict)