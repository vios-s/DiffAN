
from logging import raiseExceptions
import torch
import pandas as pd
import numpy as np
import networkx as nx
from functorch import vmap, jacrev, grad
import cdt
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
import gc
from diffan.gaussian_diffusion import GaussianDiffusion, UniformSampler, get_named_beta_schedule, mean_flat, \
                                         LossType, ModelMeanType, ModelVarType
from diffan.nn import DiffMLP
from diffan.pruning import cam_pruning
from diffan.utils import full_DAG


class DiffANM_greedy():
    def __init__(self, n_nodes, epochs: int = int(1e4), 
                batch_size : int = 1024, learning_rate : float = 0.001, 
                dropout_type = "double_random"):
        self.n_nodes = n_nodes
        assert self.n_nodes > 1, "Not enough nodes, make sure the dataset contain at least 2 variables (columns)."
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ## Diffusion parameters
        self.n_steps = int(1e2)
        betas = get_named_beta_schedule(schedule_name = "linear", num_diffusion_timesteps = self.n_steps, scale = 1, beta_start = 0.0001, beta_end = 0.02)
        self.gaussian_diffusion = GaussianDiffusion(betas = betas, 
                                                    loss_type = LossType.MSE, 
                                                    model_mean_type= ModelMeanType.EPSILON,#START_X,EPSILON
                                                    model_var_type=ModelVarType.FIXED_LARGE,
                                                    rescale_timesteps = True,
                                                    )
        self.schedule_sampler = UniformSampler(self.gaussian_diffusion)

        ## Diffusion training
        self.model = DiffMLP(n_nodes).to(self.device)
        self.epochs = epochs 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_wait = 300 #int(1e3) #in epochs 500

        ## Topological Ordering
        self.n_votes = 10
        ## Pruning
        self.cutoff = 0.001
    
    def fit(self, X):
        X = (X - X.mean(0, keepdims = True)) / X.std(0, keepdims = True)
        X = torch.FloatTensor(X).to(self.device)
        order = self.topological_ordering(X)
        out_dag = self.pruning(order, X.detach().cpu().numpy())
        return out_dag, order

    def pruning(self, order, X):
        return cam_pruning(full_DAG(order), X, self.cutoff)

    def train_score(self, X):
        n_samples, n_nodes = X.shape
        self.best_loss = float("inf")
        self.model = DiffMLP(n_nodes).to(self.device)
        self.model.float()
        self.opt = torch.optim.Adam(self.model.parameters(), self.learning_rate)

        self.model.train()
        best_model_state_epoch = 300
       
        val_ratio = 0.2
        val_size = int(n_samples * val_ratio)
        train_size = n_samples - val_size
        X_train, X_val = X[:train_size],X[train_size:]
        #X_train, X_val = torch.utils.data.random_split(X,[train_size,val_size])#, generator=torch.Generator(device = X.device).manual_seed(42)
        data_loader_val = torch.utils.data.DataLoader(X_val, min(val_size, self.batch_size))
        data_loader = torch.utils.data.DataLoader(X_train, min(train_size, self.batch_size), drop_last= True)# , shuffle = True
        pbar = tqdm(range(self.epochs), desc = "Training Epoch")
        for epoch in pbar:
            loss_per_step = []
            for steps, x_start in enumerate(data_loader):
                # apply noising and masking
                x_start = x_start.float().to(self.device)
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                noise = torch.randn_like(x_start).to(self.device)
                x_t = self.gaussian_diffusion.q_sample(x_start, t, noise=noise)
                # get loss function
                model_output = self.model(x_t, self.gaussian_diffusion._scale_timesteps(t))
                diffusion_losses = (noise - model_output) ** 2
                diffusion_loss = (diffusion_losses.mean(dim=list(range(1, len(diffusion_losses.shape)))) * weights).mean()
                loss_per_step.append(diffusion_loss.item())
                self.opt.zero_grad()
                diffusion_loss.backward()
                self.opt.step()

            if epoch % 10 == 0 and epoch > best_model_state_epoch:
                with torch.no_grad():
                    loss_per_step_val = []
                    for steps, x_start in enumerate(data_loader_val):
                        t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                        noise = torch.randn_like(x_start).to(self.device)
                        x_t = self.gaussian_diffusion.q_sample(x_start, t, noise=noise)
                        model_output = self.model(x_t, self.gaussian_diffusion._scale_timesteps(t))
                        diffusion_losses = (noise - model_output) ** 2
                        diffusion_loss = (diffusion_losses.mean(dim=list(range(1, len(diffusion_losses.shape)))) * weights).mean()
                        loss_per_step_val.append(diffusion_loss.item())
                    epoch_val_loss = np.mean(loss_per_step_val)

                    if self.best_loss > epoch_val_loss:
                        self.best_loss = epoch_val_loss
                        best_model_state = deepcopy(self.model.state_dict())
                        best_model_state_epoch = epoch
                pbar.set_postfix({'Epoch Loss': epoch_val_loss})
            
            if epoch - best_model_state_epoch > self.early_stopping_wait: # Early stopping
                break
        print(f"Early stoping at epoch {epoch}")
        print(f"Best model at epoch {best_model_state_epoch} with loss {self.best_loss}")
        self.model.load_state_dict(best_model_state)

    def topological_ordering(self, X, step = None, eval_batch_size = None):
        n_samples = X.shape[0]
        self.batch_size = min(n_samples, self.batch_size)
        if eval_batch_size is None:
            eval_batch_size = self.batch_size
        eval_batch_size = min(eval_batch_size, X.shape[0])
        
        order = []
        active_nodes = list(range(self.n_nodes))
        pbar = tqdm(range(self.n_nodes-1), desc = "Nodes ordered ")

        for jac_step in pbar:
            X_in = X[:, active_nodes]
            self.train_score(X_in)
            print("Model trained")
            self.model.eval()
            data_loader = torch.utils.data.DataLoader(X_in, eval_batch_size, drop_last = True)#shuffle = True, 
            if step is None:
                leaves = []
                for steps in range(0, self.n_steps+1, self.n_steps//self.n_votes):
                    jacobians = []
                    for x_start in data_loader:
                        t_functorch = (torch.ones(1)*steps).long().to(self.device) # test if other ts or random ts are better, self.n_steps
                        # active_nodes selects which outputs are propagating gradient
                        model_fn_functorch = lambda x: self.model(x, self.gaussian_diffusion._scale_timesteps(t_functorch))
                        # active_nodes selects for which inputs the jacobian is computed 
                        x_start = x_start.float().to(self.device)
                        jacobian = vmap(jacrev(model_fn_functorch))(x_start.unsqueeze(1)).squeeze()
                        jacobian_np = jacobian.detach().cpu().numpy()
                        del jacobian
                        jacobians.append(jacobian_np)
                        if len(jacobians)*eval_batch_size >= self.batch_size:
                            break
                    jacobian_var_diag = np.concatenate(jacobians, 0).var(0).diagonal()
                    var_sorted_nodes = np.argsort(jacobian_var_diag)
                    leaf_per_step = var_sorted_nodes[0]
                    leaves.append(leaf_per_step)
                #leaf_np = np.bincount(leaves).argmax()
                leaf = Counter(leaves).most_common(1)[0][0]
                #assert leaf_np == leaf
                
            
            order.append(active_nodes[leaf])
            active_nodes.pop(leaf)
            pbar.set_postfix({'Node ordered': order[-1]})

        order.append(active_nodes[0])
        order.reverse()
        return order
