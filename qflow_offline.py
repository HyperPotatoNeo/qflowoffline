import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import d4rl
import random
import argparse
from distutils.util import strtobool
import os
import time
from model import DiffusionModel, QFlow
from IQL_PyTorch.src.value_functions import TwinQ
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class D4RLDataset(Dataset):
    def __init__(self, data):
        self.states = data['observations']
        self.actions = data['actions']

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        return states, actions
    
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="advantage-diffusion",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='swish',
        help="the entity (team) of wandb's project")

    parser.add_argument("--env-id", type=str, default="hopper-medium-expert-v2",
        help="the id of the environment")
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--sample-freq", type=int, default=1)
    parser.add_argument("--predict", type=str, default='epsilon')
    parser.add_argument("--policy-net", type=str, default='mlp')
    parser.add_argument("--num-eval", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--extra", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Extra sampling steps")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    filename = args.env_id+"_"+args.exp_name
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    env = gym.make(args.env_id)
    dataset = env.get_dataset()
    #if args.predict == 'epsilon':
    dataset['actions'] = np.arctanh(np.clip(dataset['actions'],-0.99,0.99))
    data = D4RLDataset(dataset)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    
    bc_model = DiffusionModel(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], diffusion_steps=args.diffusion_steps, predict=args.predict, policy_net=args.policy_net).to(device)
    bc_model.load_state_dict(torch.load('bc_models/'+args.env_id+'_'+'train_bc.pth'))
    q = TwinQ(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    q.load_state_dict(torch.load('q_models/'+args.env_id+'_qf.pth'))
    
    qflow = QFlow(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], diffusion_steps=args.diffusion_steps, predict=args.predict, q_net=q, bc_net=bc_model, alpha=args.alpha).to(device)
    optimizer = torch.optim.Adam(list(qflow.qflow.out_model.parameters()) + list(qflow.qflow.means_scaling_model.parameters()) + list(qflow.qflow.x_model.parameters()), lr=args.lr)
    
    global_step = 0
    for epoch in range(args.n_epochs):
        for states, actions in dataloader:
            if global_step % args.sample_freq == 0:
                optimizer.zero_grad()
                states = states.to(device)
                loss, logZSample = qflow.compute_loss(states)
                loss.backward()
                optimizer.step()
                sample_loss = loss.item()

            states = states.to(device)
            actions = actions.to(device)
            loss, logC = qflow.compute_loss_with_sample(states, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()

            if global_step%5 == 0 and args.track:
                writer.add_scalar("loss/sample_loss", sample_loss, global_step)
                writer.add_scalar("loss/batch_loss", batch_loss, global_step)
                writer.add_scalar("loss/logZSample", logZSample, global_step)
                writer.add_scalar("loss/logC", logC, global_step)
            with torch.no_grad():
                if ((global_step+1)%500) == 0:
                    avg_reward = 0.0
                    for i in range(args.num_eval):
                        s = env.reset()
                        steps = 0
                        done = False
                        while not done:
                            steps+=1
                            s_tensor = torch.tensor(s).float().to(device).unsqueeze(0)
                            a, _, _ = qflow.sample(s_tensor, extra=args.extra)#.detach().cpu().numpy()
                            a = torch.tanh(torch.tensor(a)).detach().cpu().numpy()[0]
                            s, r, done, _ = env.step(a)
                            avg_reward += r
                            #print(steps, s[:2])
                    avg_reward /= args.num_eval
                    if not 'antmaze' in args.env_id:
                        avg_reward = env.get_normalized_score(avg_reward)*100
                    print('AVG REWARD:', avg_reward)
                    writer.add_scalar("eval/avg_reward", avg_reward, global_step)
            global_step += 1