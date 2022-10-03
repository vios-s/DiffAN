
from logging import raiseExceptions
import torch
import numpy as np
from functorch import vmap, jacrev, jacfwd
from collections import Counter
from copy import deepcopy
from tqdm import tqdm

from diffan.gaussian_diffusion import GaussianDiffusion, UniformSampler, get_named_beta_schedule, mean_flat, \
                                         LossType, ModelMeanType, ModelVarType
from diffan.nn import DiffMLP
from diffan.pruning import cam_pruning
from diffan.utils import full_DAG


class DiffAN():
    def __init__(self, n_nodes, masking = True, residue= True, 
                epochs: int = int(3e3), batch_size : int = 1024, learning_rate : float = 0.001):
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
        self.epochs = epochs 
        self.batch_size = batch_size
        self.model = DiffMLP(n_nodes).to(self.device)
        self.model.float()
        self.opt = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.val_diffusion_loss = []
        self.best_loss = float("inf")
        self.early_stopping_wait = 300

        ## Topological Ordering
        self.n_votes = 3
        self.masking = masking
        self.residue = residue
        self.sorting = (not masking) and (not residue)
        ## Pruning
        self.cutoff = 0.001
    
    def fit(self, X):
        X = (X - X.mean(0, keepdims = True)) / X.std(0, keepdims = True)
        X = torch.FloatTensor(X).to(self.device)
        self.train_score(X)
        order = self.topological_ordering(X)
        out_dag = self.pruning(order, X.detach().cpu().numpy())
        return out_dag, order

    def pruning(self, order, X):
        return cam_pruning(full_DAG(order), X, self.cutoff)
    
    def train_score(self, X, fixed = None):
        if fixed is not None:
            self.epochs = fixed
        best_model_state_epoch = 300
        self.model.train()
        n_samples = X.shape[0]
        self.batch_size = min(n_samples, self.batch_size)
        val_ratio = 0.2
        val_size = int(n_samples * val_ratio)
        train_size = n_samples - val_size 
        X = X.to(self.device)
        X_train, X_val = X[:train_size],X[train_size:]
        data_loader_val = torch.utils.data.DataLoader(X_val, min(val_size, self.batch_size))
        data_loader = torch.utils.data.DataLoader(X_train, min(train_size, self.batch_size), drop_last= True)
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
            if fixed is None:
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
        if fixed is None:
            print(f"Early stoping at epoch {epoch}")
            print(f"Best model at epoch {best_model_state_epoch} with loss {self.best_loss}")
            self.model.load_state_dict(best_model_state)    

    def topological_ordering(self, X, step = None, eval_batch_size = None):
        
        if eval_batch_size is None:
            eval_batch_size = self.batch_size
        eval_batch_size = min(eval_batch_size, X.shape[0])

        X = X[:self.batch_size]
        
        self.model.eval()
        order = []
        active_nodes = list(range(self.n_nodes))
        
        
        steps_list = [step] if step is not None else range(0, self.n_steps+1, self.n_steps//self.n_votes)
        if self.sorting:
            steps_list = [self.n_steps//2]
        pbar = tqdm(range(self.n_nodes-1), desc = "Nodes ordered ")
        leaf = None
        for jac_step in pbar:        
            leaves = []
            for i, steps in enumerate(steps_list):
                data_loader = torch.utils.data.DataLoader(X, eval_batch_size, drop_last = True)

                model_fn_functorch = self.get_model_function_with_residue(steps, active_nodes, order)
                leaf_ = self.compute_jacobian_and_get_leaf(data_loader, active_nodes, model_fn_functorch)
                if self.sorting:
                    order = leaf_.tolist()
                    order.reverse()
                    return order
                leaves.append(leaf_)

            leaf = Counter(leaves).most_common(1)[0][0]
            leaf_global = active_nodes[leaf]
            order.append(leaf_global)
            active_nodes.pop(leaf)


        order.append(active_nodes[0])
        order.reverse()

        return order

    def get_model_function_with_residue(self, step, active_nodes, order):
        t_functorch = (torch.ones(1)*step).long().to(self.device) # test if other ts or random ts are better, self.n_steps
        get_score_active = lambda x: self.model(x, self.gaussian_diffusion._scale_timesteps(t_functorch))[:,active_nodes]
        get_score_previous_leaves = lambda x: self.model(x, self.gaussian_diffusion._scale_timesteps(t_functorch))[:,order]
        def model_fn_functorch(X):
            score_active = get_score_active(X).squeeze()

            if self.residue and len(order) > 0:
                score_previous_leaves = get_score_previous_leaves(X).squeeze()
                jacobian_ = jacfwd(get_score_previous_leaves)(X).squeeze()
                if len(order) == 1:
                    jacobian_, score_previous_leaves = jacobian_.unsqueeze(0), score_previous_leaves.unsqueeze(0)
                score_active += torch.einsum("i,ij -> j",score_previous_leaves/ jacobian_[:, order].diag(),jacobian_[:, active_nodes])#

            return score_active
        return model_fn_functorch

    def get_masked(self, x, active_nodes):
        dropout_mask = torch.zeros_like(x).to(self.device)
        dropout_mask[:, active_nodes] = 1
        return (x * dropout_mask).float()
    
    def compute_jacobian_and_get_leaf(self, data_loader, active_nodes, model_fn_functorch):
        jacobian = []
        for x_batch in data_loader:
            x_batch_dropped = self.get_masked(x_batch, active_nodes) if self.masking else x_batch
            jacobian_ = vmap(jacrev(model_fn_functorch))(x_batch_dropped.unsqueeze(1)).squeeze()
            jacobian.append(jacobian_[...,active_nodes].detach().cpu().numpy())
        jacobian = np.concatenate(jacobian, 0)
        leaf = self.get_leaf(jacobian)
        return leaf
    
    def get_leaf(self, jacobian_active):
        jacobian_var = jacobian_active.var(0)
        jacobian_var_diag = jacobian_var.diagonal()
        var_sorted_nodes = np.argsort(jacobian_var_diag)
        if self.sorting:
            return var_sorted_nodes
        leaf_current = var_sorted_nodes[0]
        return leaf_current
