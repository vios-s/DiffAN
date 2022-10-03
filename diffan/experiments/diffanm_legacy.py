
from logging import raiseExceptions
import torch
import pandas as pd
import numpy as np
import networkx as nx
from functorch import vmap, jacrev, jacfwd
import cdt
from collections import Counter
from copy import deepcopy
from tqdm import tqdm

from diffan.gaussian_diffusion import GaussianDiffusion, UniformSampler, get_named_beta_schedule, mean_flat, \
                                         LossType, ModelMeanType, ModelVarType
from diffan.nn import DiffMLP
from diffan.pruning import cam_pruning
from diffan.utils import full_DAG


class DiffANM():
    def __init__(self, n_nodes, epochs: int = int(3e3), 
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
        self.epochs = epochs 
        self.batch_size = batch_size
        self.model = DiffMLP(n_nodes).to(self.device)
        self.model.float()
        self.opt = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.val_diffusion_loss = []
        self.best_loss = float("inf")
        self.early_stopping_wait = 300 #int(1e3) #in epochs 500

        self.dropout_type = dropout_type 
        assert self.dropout_type in ["const","random","single_random","double_random"]

        ## Topological Ordering
        self.n_votes = 3
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
        X_train, X_val = X[:train_size],X[train_size:]#torch.utils.data.random_split(X, [train_size,val_size], generator=torch.Generator(device = self.device).manual_seed(42))#
        data_loader_val = torch.utils.data.DataLoader(X_val, min(val_size, self.batch_size))
        data_loader = torch.utils.data.DataLoader(X_train, min(train_size, self.batch_size), drop_last= True)#shuffle = True,
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

    def train_score_old(self, X, val_type = "drop"):
        n_samples = X.shape[0]
        self.batch_size = min(n_samples, self.batch_size)
        self.val_diffusion_loss = []
        
        data_loader = torch.utils.data.DataLoader(X, self.batch_size, shuffle = True, drop_last= True)
        nodes_list = list(range(self.n_nodes))
        nodes_list_val = list(range(self.n_nodes))
        pbar = tqdm(range(self.epochs), desc = "Training Epoch")
        for epoch in pbar:
            for steps, x_start in enumerate(data_loader):
                dropout_mask = self.get_training_mask(x_start.shape,nodes_list)
                # apply noising and masking
                x_start = x_start.float().to(self.device)
                t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                noise = torch.randn_like(x_start).to(self.device)
                x_t = self.gaussian_diffusion.q_sample(x_start, t, noise=noise)
                x_t_dropped = x_t * dropout_mask
                # get loss function
                model_output = self.model(x_t_dropped, self.gaussian_diffusion._scale_timesteps(t))
                diffusion_losses = (noise - model_output) ** 2
                diffusion_losses_dropped = diffusion_losses * dropout_mask
                diffusion_loss = (diffusion_losses_dropped.mean(dim=list(range(1, len(diffusion_losses_dropped.shape)))) * weights).mean()
                
                self.opt.zero_grad()
                diffusion_loss.backward()
                self.opt.step()
                
            if epoch % 10 == 0:
                with torch.no_grad():

                    t, weights = self.schedule_sampler.sample(x_start.shape[0], self.device)
                    noise = torch.randn_like(x_start).to(self.device)
                    x_t = self.gaussian_diffusion.q_sample(x_start, t, noise=noise)

                    if val_type == "drop":
                        diffusion_loss_per_drop = []
                        # same noise at each iteration
                        intervals = 10
                        interval = (self.n_nodes-1)//intervals if self.n_nodes > intervals+1 else 1
                        for n_nodes_to_drop in range(0, self.n_nodes-1, interval):
                            dropout_mask = torch.ones_like(x_start).to(self.device)
                            variables_to_drop = nodes_list_val[:n_nodes_to_drop] if n_nodes_to_drop > 0 else []
                            dropout_mask[:, variables_to_drop] = 0
                            x_t_dropped = x_t * dropout_mask
                            model_output = self.model(x_t_dropped, self.gaussian_diffusion._scale_timesteps(t))
                            diffusion_losses = (noise - model_output) ** 2
                            diffusion_losses_dropped = diffusion_losses * dropout_mask
                            diffusion_loss = (diffusion_losses_dropped.mean(dim=list(range(1, len(diffusion_losses_dropped.shape)))) * weights).mean()
                            diffusion_loss_per_drop.append(diffusion_loss.item())
                        self.val_diffusion_loss.append(np.array(diffusion_loss_per_drop))
                        epoch_val_loss = np.array(diffusion_loss_per_drop).sum()
                    elif val_type == "const":
                        model_output = self.model(x_t, self.gaussian_diffusion._scale_timesteps(t))
                        diffusion_losses = (noise - model_output) ** 2
                        diffusion_loss = (diffusion_losses.mean(dim=list(range(1, len(diffusion_losses.shape)))) * weights).mean()
                        self.val_diffusion_loss.append(np.array(diffusion_loss.item()))
                        epoch_val_loss = diffusion_loss.item()
                    else:
                        raise Exception("Wrong val_type")

                    if self.best_loss > epoch_val_loss:
                        self.best_loss = epoch_val_loss
                        best_model_state = deepcopy(self.model.state_dict())
                        best_model_state_epoch = epoch

            pbar.set_postfix({'Val Loss': epoch_val_loss})
            if epoch > 500 and epoch - best_model_state_epoch > self.early_stopping_wait: # Early stopping
                break
        print(f"Early stoping at epoch {epoch}")
        print(f"Best model at epoch {best_model_state_epoch} with loss {self.best_loss}")
        self.model.load_state_dict(best_model_state)
    
    def get_training_mask(self, shape, nodes_list):
        # get dropping mask
        dropout_mask = torch.ones(shape).to(self.device)
        np.random.shuffle(nodes_list)
        if self.dropout_type == "const":
            pass
        elif self.dropout_type == "random":
            dropout_mask = (torch.randn(*shape) > 0).to(torch.long).to(self.device)
        elif self.dropout_type == "single_random":
            variables_to_drop = nodes_list[:1]
            dropout_mask[:, variables_to_drop] = 0
        elif self.dropout_type == "double_random":
            tau_n_nodes_to_drop = np.random.randint(low = 0, high=self.n_nodes-2)
            variables_to_drop = nodes_list[:tau_n_nodes_to_drop] if tau_n_nodes_to_drop > 0 else []
            dropout_mask[:, variables_to_drop] = 0
        return dropout_mask.float()

    def topological_ordering(self, X, step = None, eval_batch_size = None):
        if eval_batch_size is None:
            eval_batch_size = self.batch_size
        eval_batch_size = min(eval_batch_size, X.shape[0])
        
        #X = X.to(self.device).double()
        X = X[:self.batch_size]
        self.model.eval()
        order = []
        data_loader = torch.utils.data.DataLoader(X, eval_batch_size, drop_last = True)#, shuffle = True, drop_last = True, generator=torch.Generator(device = self.device))
        active_nodes = list(range(self.n_nodes))
        pbar = tqdm(range(self.n_nodes-1), desc = "Nodes ordered ")
        sorteds = []
        jacobian_vars = []
        for jac_step in pbar:

            dropout_mask = torch.zeros((eval_batch_size,self.n_nodes)).to(self.device)
            dropout_mask[:, active_nodes] = 1
            
            if step is None:
                leaves = []
                for steps in range(0, self.n_steps+1, self.n_steps//self.n_votes):
                    jacobians = []
                    for x_start in data_loader:
                        x_start_dropped = (x_start * dropout_mask).float()
                        t_functorch = (torch.ones(1)*steps).long().to(self.device) # test if other ts or random ts are better, self.n_steps
                        # active_nodes selects which outputs are propagating gradient
                        model_fn_functorch = lambda x: self.model(x, self.gaussian_diffusion._scale_timesteps(t_functorch))[:,active_nodes]
                        # active_nodes selects for which inputs the jacobian is computed 
                        jacobian = vmap(jacrev(model_fn_functorch))(x_start_dropped.unsqueeze(1)).squeeze()
                        jacobian_np = jacobian[...,active_nodes].detach().cpu().numpy()
                        jacobians.append(jacobian_np)
                        if len(jacobians)*eval_batch_size >= self.batch_size:
                            break
                    jacobian_var_diag = np.concatenate(jacobians, 0).var(0).diagonal()
                    var_sorted_nodes = np.argsort(jacobian_var_diag)
                    sorteds.append([active_nodes[i] for i in var_sorted_nodes])
                    leaf_per_step = var_sorted_nodes[0]
                    leaves.append(leaf_per_step)

                leaf = Counter(leaves).most_common(1)[0][0]
            else:
                t_functorch = (torch.ones(1)*step).long().to(self.device) # test if other ts or random ts are better, self.n_steps
                def model_fn_functorch(X):
                    score_ = self.model(X, self.gaussian_diffusion._scale_timesteps(t_functorch))
                    return score_[:,active_nodes]

                # active_nodes selects which outputs are propagating gradient
                jacobians = []
                for x_start in data_loader:
                    x_start_dropped = (x_start * dropout_mask).float()
                    # active_nodes selects for which inputs the jacobian is computed 
                    jacobian = vmap(jacrev(model_fn_functorch))(x_start_dropped.unsqueeze(1)).squeeze()
                    jacobian_np = jacobian[...,active_nodes].detach().cpu().numpy()
                    jacobians.append(jacobian_np)
                jacobian_var = np.concatenate(jacobians, 0).var(0)
                jacobian_var_diag = jacobian_var.diagonal()
                var_sorted_nodes = np.argsort(jacobian_var_diag)
                sorteds.append(([active_nodes[i] for i in var_sorted_nodes],[jacobian_var_diag[i] for i in var_sorted_nodes]))
                jacobian_vars.append(jacobian_var)
                leaf = var_sorted_nodes[0]
            
            order.append(active_nodes[leaf])
            active_nodes.pop(leaf)
            
        
        order.append(active_nodes[0])
        order.reverse()
        
        if step is None:
            return order
        else:
            return order, sorteds, jacobian_vars


    def topological_ordering_correction(self, X, step = None, eval_batch_size = None, use_residue = True):
        
        if eval_batch_size is None:
            eval_batch_size = self.batch_size
        eval_batch_size = min(eval_batch_size, X.shape[0])

        X = X[:self.batch_size]
        
        self.model.eval()
        order = []
        active_nodes = list(range(self.n_nodes))
        
        if step is not None:
            residue = torch.zeros((1,self.batch_size,self.n_nodes)).to(self.device)
            steps_list = [step]
        else:
            residue = torch.zeros((self.n_votes+1,self.batch_size,self.n_nodes)).to(self.device)
            steps_list = range(0, self.n_steps+1, self.n_steps//self.n_votes)
        

        pbar = tqdm(range(self.n_nodes-1), desc = "Nodes ordered ")
        leaf = None
        for jac_step in pbar:        
            leaves = []
            for i, steps in enumerate(steps_list):

                data_loader = torch.utils.data.DataLoader(torch.stack((X, residue[i]), dim=-1), eval_batch_size, drop_last = True)

                #model_fn_functorch = self.get_model_function(steps, active_nodes)
                #leaf_ = self.compute_jacobian_and_get_leaf(data_loader, active_nodes, model_fn_functorch)
                model_fn_functorch = self.get_model_function_with_residue(steps, active_nodes, leaf)
                leaf_ = self.compute_jacobian_and_get_leaf_direct(data_loader, active_nodes, model_fn_functorch)
                
                
                leaves.append(leaf_)

            leaf = Counter(leaves).most_common(1)[0][0]
            leaf_global = active_nodes[leaf]
            order.append(leaf_global)
            active_nodes.pop(leaf)
            
            if use_residue:
                for i, steps in enumerate(steps_list):

                    data_loader = torch.utils.data.DataLoader(torch.stack((X, residue[i]), dim=-1), eval_batch_size, drop_last = True)

                    model_fn_functorch = self.get_model_function(steps, active_nodes, masked = False)

                    residue[i] += self.get_residue(data_loader, active_nodes, model_fn_functorch, leaf_global)

        order.append(active_nodes[0])
        order.reverse()

        return order

    def get_model_function(self, step, active_nodes, masked = True):
        t_functorch = (torch.ones(1)*step).long().to(self.device) # test if other ts or random ts are better, self.n_steps
        def model_fn_functorch(X, residue_term_):
            score_ = self.model(X, self.gaussian_diffusion._scale_timesteps(t_functorch))
            new_score = score_ + residue_term_
            return new_score[:,active_nodes] if masked else new_score
        return model_fn_functorch    

    def get_model_function_with_residue(self, step, active_nodes, leaf, masked = True):
        t_functorch = (torch.ones(1)*step).long().to(self.device) # test if other ts or random ts are better, self.n_steps
        get_score = lambda x: self.model(x, self.gaussian_diffusion._scale_timesteps(t_functorch))
        def model_fn_functorch(X):
            score_ = get_score(X).squeeze()
            if leaf is not None:
                jacobian_ = jacrev(get_score)(X).squeeze()
                residue_ =  jacobian_[leaf,:] * score_[leaf] #/ jacobian_[leaf, leaf]
                new_score = score_ + residue_
            else:
                new_score = score_
            return new_score[active_nodes] if masked else new_score
        return model_fn_functorch         

   
    def get_masked(self, x, active_nodes):
        dropout_mask = torch.zeros_like(x).to(self.device)
        dropout_mask[:, active_nodes] = 1
        return (x * dropout_mask).float()
    
    def get_residue(self, data_loader, active_nodes, model_fn_functorch, leaf):
        score = []
        jacobian = []
        for batch in data_loader:
            x_batch, previous_residue = batch[...,0], batch[...,-1]
            x_batch_dropped = self.get_masked(x_batch, active_nodes)
            score_ = vmap(model_fn_functorch)(x_batch_dropped.unsqueeze(1), previous_residue.unsqueeze(1)).squeeze()
            score.append(score_.detach().cpu())
            jacobian_ = vmap(jacrev(model_fn_functorch))(x_batch_dropped.unsqueeze(1), previous_residue.unsqueeze(1)).squeeze()
            jacobian.append(jacobian_.detach().cpu())
        score = torch.cat(score,0)
        jacobian = torch.cat(jacobian,0)
        residue = torch.einsum('i,ij->ij', 
                    score[:,leaf]/jacobian[:,leaf, leaf], 
                    jacobian[:,leaf,:]).to(self.device)
        return residue

    def compute_jacobian_and_get_leaf(self, data_loader, active_nodes, model_fn_functorch):
        jacobian = []
        for batch in data_loader:
            x_batch, previous_residue = batch[...,0], batch[...,-1]
            x_batch_dropped = self.get_masked(x_batch, active_nodes)
            jacobian_ = vmap(jacrev(model_fn_functorch))(x_batch_dropped.unsqueeze(1), previous_residue.unsqueeze(1)).squeeze()
            jacobian.append(jacobian_[...,active_nodes].detach().cpu().numpy())
        jacobian = np.concatenate(jacobian, 0)
        leaf = self.get_leaf(jacobian)
        return leaf

    
    def compute_jacobian_and_get_leaf_direct(self, data_loader, active_nodes, model_fn_functorch):
        jacobian = []
        for batch in data_loader:
            x_batch, previous_residue = batch[...,0], batch[...,-1]
            x_batch_dropped = self.get_masked(x_batch, active_nodes)
            jacobian_ = vmap(jacrev(model_fn_functorch))(x_batch_dropped.unsqueeze(1)).squeeze()
            jacobian.append(jacobian_[...,active_nodes].detach().cpu().numpy())
        jacobian = np.concatenate(jacobian, 0)
        leaf = self.get_leaf(jacobian)
        return leaf
    
    def get_leaf(self, jacobian_active):
        jacobian_var = jacobian_active.var(0)
        jacobian_var_diag = jacobian_var.diagonal()
        var_sorted_nodes = np.argsort(jacobian_var_diag)
        leaf_current = var_sorted_nodes[0]
        return leaf_current
