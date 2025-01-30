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


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_action_log_probs(distributions, state_actions):
    return distributions.log_prob(state_actions).sum(dim=1)[:, None]


class SoftActorCritic(Algorithm):

    def __init__(self, environment_helper, agent):
        super().__init__(environment_helper, agent)
        hard_update(self.agent.networks['target_critic'], self.agent.networks['online_critic'])
        self.alpha = environment_helper.run.sac_config.alpha

    def train(self, memory: TensorDict, update_count: int):
        run: Run = self.environment_helper.run
        batch_size = run.training_config.batch_size
        memory = memory.view(-1)
        batches_per_timestep = 1
        losses = [[] for i in range(5)]
        idx = torch.randperm(len(memory))
        shuffled_memory = torch.clone(memory)[idx]
        shuffled_memory['reward'] = shuffled_memory['reward'] - shuffled_memory['reward'].mean()
        shuffled_memory['reward'] = (shuffled_memory['reward'] / shuffled_memory['reward'].std())
        for i in range(batches_per_timestep):
            batch = shuffled_memory[int(i * batch_size):int((i + 1) * batch_size)]
            state_batch, next_state_batch, reward_batch, action_batch, mask_batch = batch[
                'current_state'], batch['next_state'], batch['reward'], batch['action'], batch[
                    'terminated']
            with torch.no_grad():
                next_state_actions, distributions = self.agent.act(next_state_batch,
                                                                   return_dist=True)
                next_state_actions_log_prob = get_action_log_probs(distributions,
                                                                   next_state_actions)
                qf1_next_target, qf2_next_target = self.agent.networks['target_critic'](
                    next_state_batch, next_state_actions.to(run.device))
                min_qf_next_target = torch.min(
                    qf1_next_target, qf2_next_target) - self.alpha * next_state_actions_log_prob
                next_q_value = (reward_batch + mask_batch * run.sac_config.gamma *
                                (min_qf_next_target)).to(run.dtype)
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
            torch.nn.utils.clip_grad_norm_(self.agent.networks['online_critic'].parameters(),
                                           Run.instance().ppo_config.max_grad_norm)
            self.agent.optimizers['online_critic'].step()

            state_actions, distributions = self.agent.act(state_batch, return_dist=True)

            qf1_pi, qf2_pi = self.agent.networks['online_critic'](state_batch,
                                                                  state_actions.to(run.device))
            actions_log_prob = get_action_log_probs(distributions, state_actions)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = (
                (self.alpha * actions_log_prob) -
                min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.agent.optimizers['actor'].zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.networks['actor'].parameters(),
                                           Run.instance().ppo_config.max_grad_norm)
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

                self.alpha = self.agent.log_alpha.exp()
                alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(run.device)
                alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

            if update_count % run.sac_config.target_update_interval == 0:
                soft_update(self.agent.networks['target_critic'],
                            self.agent.networks['online_critic'], run.sac_config.tau)
            losses[0].append(qf1_loss.item())
            losses[1].append(qf2_loss.item())
            losses[2].append(policy_loss.item())
            losses[3].append(min_qf_pi.mean().item())
            losses[4].append(alpha_loss.item())
        return (sum(losses[j]) / len(losses[j]) for j in range(5))

    def _iterate(self):
        self.environment_helper.reset(release_memory=False)
        self.environment_helper.reset_environment(test_phase=False)
        next_state = self.environment_helper.get_state(test_phase=False)
        batch_size = len(next_state)
        run = self.environment_helper.run
        device = run.device
        sub_memory = []
        update_count = 0
        losses = []
        trained = False
        for timestep in range(run.environment_config.maximum_timesteps):
            # 1. act
            with torch.no_grad():
                current_state = torch.clone(next_state)
                action, distributions = self.agent.act(current_state,
                                                       return_dist=True,
                                                       test_phase=False)
            train_interval = 5  #int(run.environment_config.maximum_timesteps /
            #   (run.sac_config.target_update_interval * 2))
            # 2. train
            if len(self.environment_helper.memory +
                   sub_memory) > run.training_config.batch_size and timestep % train_interval == 0:
                trained = True
                losses.append(
                    list(
                        self.train(torch.cat(self.environment_helper.memory + sub_memory, dim=1),
                                   update_count)))
                update_count += 1
            # 3. add memory item
            self.environment_helper.step(action)
            next_state = self.environment_helper.get_state(test_phase=False)
            memory_item = {
                'current_state':
                current_state.unsqueeze(1).detach(),
                'action':
                action.unsqueeze(1).detach(),
                'reward':
                torch.tensor(
                    self.environment_helper.timestep.reward)[:,
                                                             None].to(device).unsqueeze(1).detach(),
                'next_state':
                next_state.unsqueeze(1).detach(),
                'terminated':
                ~torch.tensor(self.environment_helper.timestep.terminated[:, None]).to(
                    device).unsqueeze(1).detach()
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
        if len(losses) > 0:
            Logger.log(
                f"qf1_loss: {torch.tensor(losses)[:,0].mean()} , qf2_loss: {torch.tensor(losses)[:,1].mean()}",
                episode=run.dynamic_config.current_episode,
                log_type=Logger.REWARD_TYPE,
                print_message=True)
            Logger.log(
                f"policy_loss: {torch.tensor(losses)[:,2].mean()}, min_qf:{torch.tensor(losses)[:,3].mean()}",
                episode=run.dynamic_config.current_episode,
                log_type=Logger.REWARD_TYPE,
                print_message=True)
            # Logger.log(
            #     f"alpha_loss: {torch.tensor(losses)[:,4].mean()} , alpha value: {self.alpha}",
            #     episode=run.dynamic_config.current_episode,
            #     log_type=Logger.REWARD_TYPE,
            #     print_message=True)
        self.environment_helper.memory = (
            sub_memory + self.environment_helper.memory)[:run.sac_config.memory_capacity]
        del sub_memory
        if trained:
            for scheduler in self.agent.schedulers.values():
                scheduler.step()
