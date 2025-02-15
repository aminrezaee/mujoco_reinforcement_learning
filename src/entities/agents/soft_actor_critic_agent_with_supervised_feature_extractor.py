from .agent import Agent
from models.transformer.with_feature_extractor.transformer_actor import TransformerActor as Actor
from models.transformer.with_feature_extractor.transformer_q_network import TransformerQNetwork as QNetwork
from models.transformer.with_feature_extractor.feature_extractor import FeatureExtractor
import torch
from entities.features import Run
from torch.optim.lr_scheduler import ExponentialLR
from os import path


class SoftActorCriticAgent(Agent):

    def initialize_networks(self):
        run = Run.instance()
        # initialize models
        feature_extractor = FeatureExtractor()
        self.networks['feature_extractor'] = feature_extractor
        self.networks['actor'] = Actor(feature_extractor)
        self.networks['online_critic'] = QNetwork(feature_extractor)
        self.networks['target_critic'] = QNetwork(feature_extractor)
        # initialize optimizers
        self.optimizers['feature_extractor'] = torch.optim.Adam(
            self.networks['feature_extractor'].parameters(), lr=run.training_config.learning_rate)
        self.optimizers['actor'] = torch.optim.Adam(self.networks['actor'].parameters(),
                                                    lr=run.training_config.learning_rate)
        self.optimizers['online_critic'] = torch.optim.Adam(
            self.networks['online_critic'].parameters(), lr=run.training_config.learning_rate)
        self.optimizers['target_critic'] = torch.optim.Adam(
            self.networks['target_critic'].parameters(), lr=run.training_config.learning_rate)
        if run.sac_config.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=run.device)
            self.optimizers['alpha'] = torch.optim.Adam(
                [self.log_alpha], lr=Run.instance().training_config.learning_rate)
        # initialize schedulers
        self.schedulers['feature_extractor'] = ExponentialLR(self.optimizers['feature_extractor'],
                                                             gamma=0.999)
        self.schedulers['actor'] = ExponentialLR(self.optimizers['actor'], gamma=0.999)
        self.schedulers['online_critic'] = ExponentialLR(self.optimizers['online_critic'],
                                                         gamma=0.999)
        self.schedulers['target_critic'] = ExponentialLR(self.optimizers['target_critic'],
                                                         gamma=0.999)
        return

    def get_state_value(self, state):
        return self.networks['critic'](state)
