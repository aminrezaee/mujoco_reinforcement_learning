import torch
from .base_algorithm import Algorithm
from entities.features import Run
from tensordict import TensorDict
import numpy as np
from torch.nn.functional import huber_loss, mse_loss
from utils.logger import Logger
from torch.nn import Module


def soft_update(target: Module, source: Module, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class SoftActorCritic(Algorithm):

    def __init__(self, environment_helper, agent):
        super().__init__(environment_helper, agent)

    def train(self, memory: TensorDict, update_count: int):
        run: Run = self.environment_helper.run
        batch_size = run.training_config.batch_size
        batches_per_epoch = int(run.environment_config.maximum_timesteps *
                                run.environment_config.num_envs / batch_size)
        memory = memory.view(-1)
        epoch_losses = [[], []]
        idx = torch.randperm(len(memory))
        shuffled_memory = memory[idx]
        for i in range(batches_per_epoch):
            batch = shuffled_memory[int(i * batch_size):int((i + 1) * batch_size)]
            state_batch, next_state_batch, reward_batch, action_batch, mask_batch = batch[
                'current_state'], batch['next_state'], batch['reward'], batch['action'], batch[
                    'terminated']
            with torch.no_grad():
                next_state_actions, distributions = self.agent.act(next_state_batch,
                                                                   return_dist=True)
                next_state_actions_log_prob = torch.cat([
                    distributions[j].log_prob(next_state_actions[j])
                    for j in range(len(next_state_actions))
                ])
                qf1_next_target, qf2_next_target = self.agent.networks['target_critic'](
                    next_state_batch, next_state_actions)
                min_qf_next_target = torch.min(
                    qf1_next_target,
                    qf2_next_target) - run.sac_config.alpha * next_state_actions_log_prob
                next_q_value = reward_batch + mask_batch * run.sac_config.gamma * (
                    min_qf_next_target)
            qf1, qf2 = self.agent.networks['online_critic'](
                state_batch, action_batch
            )  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = mse_loss(
                qf1,
                next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = mse_loss(
                qf2,
                next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss
            self.agent.optimizers['online_critic'].zero_grad()
            qf_loss.backward()
            self.agent.optimizers['online_critic'].step()

            state_actions, distributions = self.agent.act(state_batch)

            qf1_pi, qf2_pi = self.agent.networks['online_critic'](state_batch, state_actions)
            actions_log_prob = torch.cat([
                distributions[j].log_prob(state_actions[j]) for j in range(len(next_state_actions))
            ])
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = (
                (run.sac_config.alpha * actions_log_prob) -
                min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.agent.optimizers['actor'].zero_grad()
            policy_loss.backward()
            self.agent.optimizers['actor'].step()

            if run.sac_config.automatic_entropy_tuning:
                target_entropy = -torch.prod(
                    torch.Tensor(
                        self.environment_helper.get_using_environment(
                            test_phase=False).action_space.shape).to(run.device)).item()
                alpha_loss = -(self.agent.log_alpha *
                               (actions_log_prob + target_entropy).detach()).mean()

                self.agent.optimizers['alpha'].zero_grad()
                alpha_loss.backward()
                self.agent.optimizers['alpha'].step()

                run.sac_config.alpha = self.agent.log_alpha.exp()
                alpha_tlogs = run.sac_config.alpha.clone()  # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(run.device)
                alpha_tlogs = torch.tensor(run.sac_config.alpha)  # For TensorboardX logs

            if update_count % run.sac_config.target_update_interval == 0:
                soft_update(self.agent.networks['target_critic'],
                            self.agent.networks['online_critic'], run.sac_config.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(
        ), alpha_tlogs.item()

    def _iterate(self):
        self.environment_helper.reset()
        self.environment_helper.reset_environment(test_phase=False)
        next_state = self.environment_helper.get_state(test_phase=False)
        batch_size = len(next_state)
        run = self.environment_helper.run
        device = run.device
        sub_memory = []
        update_count = 0
        for _ in range(run.environment_config.maximum_timesteps):
            # 1. act
            current_state = torch.clone(next_state)
            sub_actions, distributions = self.agent.act(current_state,
                                                        return_dist=True,
                                                        test_phase=False)
            # 2. train
            if len(self.environment_helper.memory + sub_memory) > run.training_config.batch_size:
                self.train(torch.cat(self.environment_helper.memory + sub_memory, dim=1),
                           update_count)
                update_count += 1
            # 3. add memory item
            self.environment_helper.step(torch.cat(sub_actions, dim=0))
            next_state = self.environment_helper.get_state(test_phase=False)
            memory_item = {
                'current_state':
                current_state.unsqueeze(1),
                'action':
                torch.cat(sub_actions, dim=0).unsqueeze(1),
                'reward':
                torch.tensor(self.environment_helper.timestep.reward)[:,
                                                                      None].to(device).unsqueeze(1),
                'next_state':
                next_state,
                'terminated':
                torch.tensor(
                    self.environment_helper.timestep.terminated[:, None]).to(device).unsqueeze(1)
            }
            sub_memory.append(TensorDict(memory_item, batch_size=(batch_size, 1)))
        Logger.log(f"episode ended with {len(sub_memory)} timesteps",
                   episode=run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=False)
        Logger.log(f"total episode reward: {torch.cat(sub_memory, dim=1)['reward'].mean()}",
                   episode=run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        self.environment_helper.memory = self.environment_helper.memory + sub_memory
        run.dynamic_config.next_episode()
