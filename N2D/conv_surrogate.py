# -*- coding: utf-8 -*-

import torch
from ns2d import *
import pytorch_lightning as pl
import torch.optim as optim
from PIL import Image
import numpy as np
import random
from torchdiffeq import odeint
# Configure logging

import torch.nn.functional as F

class PeriodicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(PeriodicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=False)
        self.padding = padding

    def forward(self, x):
        # Gestion des conditions aux bords en "tore" (périodiques)
        if self.padding > 0:
            x = torch.cat([x[:, :, -self.padding:, :], x, x[:, :, :self.padding, :]], dim=2)  # Padding vertical
            x = torch.cat([x[:, :, :, -self.padding:], x, x[:, :, :, :self.padding]], dim=3)  # Padding horizontal
        return self.conv(x)

    

def extract_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Extrait des patchs de taille `patch_size` x `patch_size` à partir d'une image de taille N x 1 x 128 x 128.
    
    Args:
        x (torch.Tensor): Le tenseur d'entrée de forme (N, 1, 128, 128)
        patch_size (int): La taille de chaque patch (patch_size x patch_size)
    
    Returns:
        torch.Tensor: Un tenseur contenant les patchs de taille (N, patch_size * patch_size, num_patches)
    """
    # Vérifier que l'entrée est bien de taille Nx1x128x128
    assert x.shape[1] == 1 and x.shape[-2] == 128 and x.shape[-1] == 128, "L'image d'entrée doit être de taille Nx1x128x128"
    
    # Utiliser F.unfold pour extraire les patchs disjoints
    patches = torch.nn.functional.unfold(x, kernel_size=patch_size, stride=patch_size)
    
    # Reformater les patchs
    num_patches = (128 // patch_size) ** 2
    patches = patches.view(x.size(0), num_patches, -1)
    
    # Transposer pour obtenir la forme (N, patch_size * patch_size, num_patches)
    patches = patches.transpose(1, 2)
    
    return patches



        


class cnn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cnn, self).__init__()
        
        # Encoder (avec un seul downsampling)
        self.cnn = nn.Sequential(PeriodicConv2d(in_channels, 16, kernel_size=3, padding=1),
                                nn.GELU(),
                                PeriodicConv2d(16, 16, kernel_size=3, stride=1,padding =1),
                                nn.GELU(),
                                PeriodicConv2d(16, 16, kernel_size=3, stride=1,padding=1),
                                nn.Softplus(),
                                PeriodicConv2d(16, 16, kernel_size=3, stride=1,padding=1),
                                nn.Softplus(),
                                PeriodicConv2d(16, 8, kernel_size=1, stride=1),)

        self.cnn2 = nn.Sequential(PeriodicConv2d(in_channels, 16, kernel_size=5, padding=2),
                                nn.Softplus(),
                                  PeriodicConv2d(16, 16, kernel_size=5, padding=2),
                                  nn.Softplus(),
                                PeriodicConv2d(16, 8, kernel_size=1, stride=1),)


        self.cnn3 = nn.Sequential(PeriodicConv2d(16, 8, kernel_size=1, stride=1),
                                  nn.Softplus(),
                                  PeriodicConv2d(8, 1, kernel_size=1, stride=1))

        
        self.Tmax = 100.0
        self.mask_size = 2
    def forward(self, x,t):
        out_ = x
        #mask = torch.zeros(x[0].size()).to(device)
        #mask[:t] = 1
        input = torch.cat([t/self.Tmax + 0*x[:,:1],x],dim=1)
        y = self.cnn(input)#*mask)
        y2 = self.cnn2(input)
        y3 = self.cnn3(torch.cat([y,y2],dim=1))
        return y3
    


class res_cnn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(res_cnn, self).__init__()
        
        # Encoder (avec un seul downsampling)


        self.cnn = nn.Sequential(PeriodicConv2d(in_channels, 32, kernel_size=5, padding=2),
                                nn.Tanh(),
                                PeriodicConv2d(32, 32, kernel_size=5, stride=1,padding =2),
                                nn.Tanh(),
                                PeriodicConv2d(32, 32, kernel_size=3, stride=1,padding=1),
                                nn.Tanh(),
                                PeriodicConv2d(32, 32, kernel_size=3, stride=1,padding=1),
                                nn.Tanh(),
                                PeriodicConv2d(32, 1, kernel_size=3, stride=1,padding=1))

                                       


        
        self.Tmax = 100.0
        self.mask_size = 2
    def forward(self, x,t):
        out_ = x
        #mask = torch.zeros(x[0].size()).to(device)
        #mask[:t] = 1

        input = torch.cat([0*x[:,:1],x],dim=1)
        out = self.cnn(input)
        return out
    

        
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder (avec un seul downsampling)
        self.enc_conv1 = nn.Sequential(nn.Conv2d(in_channels,8,kernel_size = 1,padding=0),
                                       nn.LeakyReLU(),
                                       PeriodicConv2d(8, 16, kernel_size=5, padding=2),
                                       nn.GELU(),
                                       PeriodicConv2d(16, 16, kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       PeriodicConv2d(16, 16, kernel_size=3, padding=1))

        self.pool = nn.Sequential(PeriodicConv2d(2, 32, kernel_size=6, stride = 4,padding=1),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(32, 32, kernel_size=4, stride=4),
                                  nn.GELU(),
                                  PeriodicConv2d(32, 32, kernel_size=5, padding=2))
        
        # Decoder (upsampling)
                # Decoder (upsampling)

        
        self.up = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4),
                                nn.LeakyReLU(),
                                PeriodicConv2d(16, 16, kernel_size=3, padding=1),
                                nn.GELU(),
                                nn.ConvTranspose2d(16, 16, kernel_size=4, stride=4))
                                
        
        self.dec = nn.Sequential(PeriodicConv2d(32+2, 32, kernel_size=5, padding=2),
                                 nn.LeakyReLU(),
                                 PeriodicConv2d(32, 32, kernel_size=3, padding=1),
                                 nn.GELU(),
                                 PeriodicConv2d(32, 16, kernel_size=3, padding=1))
        
        self.out = nn.Sequential(                                 nn.GELU(),
                                 nn.Conv2d(16,out_channels,kernel_size = 1,padding=0))
        

        self.Tmax = 100.0
    def forward(self, x,t):
        n_steps = 0
        x = torch.cat([x,0*x[:,:1] + t/self.Tmax - 1],dim=1)
        #print(x.size())
        y = self.one_step(x)


        return torch.cat([x[:,:1],self.out(y)],dim=1)  # .clone() pour être explicite

    def one_step(self, x):
        # Encoder
        x1 = self.enc_conv1(x)
        x2_pooled = self.pool(x)  # Downsampling (une seule fois)

        # Decoder
        x_up = self.up(x2_pooled)  # Upsampling
        x_concat = torch.cat([x, x1, x_up], dim=1)  # Concatenation des features sans in-place operation

        x3 = self.dec(x_concat)  # Deuxième convolution dans le décodeur (output)

        # Retourne un nouveau tensor explicitement
        return x3
    

class GradientNet(pl.LightningModule):
    def __init__(self, ts, mask, lr = 1e-4,mode = "gradient",c_dim = 1, time_steps = 50, step_size = 5e-2, diff_J = True):
        super(GradientNet, self).__init__()
        self.learning_rate = lr
        self.diff_J = diff_J
        # Nombre de points dans la distribution gaussienne
        self.time_steps = time_steps
        self.step_size = step_size
        c_channels = 1
        self.c_dim = c_dim
        self.ts = ts

        self.mask = mask
        self.mode = mode

        self.reinitialize_solver = 1
        
        
                # Paramètres d'initialisation
        self.n = 128
        self.diam = 2 * torch.pi
        self.T = 0.3 #0.3
        self.max_velocity = 7
        self.viscosity = 1e-3
        self.batch_size = 150
        self.peak_wavenumber = 4
        self.scale = 1
        self.random_state = 0
        
        self.l2_cost_epoch = []
        self.grad_cost_epoch = []

        # Création du modèle
        self.ODE_model = NavierStokesFieldEvolution(
                n=self.n, diam=self.diam, T=self.T, max_velocity=self.max_velocity, viscosity=self.viscosity,
                batch_size=self.batch_size, peak_wavenumber=self.peak_wavenumber, scale=self.scale,
                random_state=self.random_state, device = self.device)
        #self.initialize_ODE_model()

        self.cond_network = UNet(2,14)
        self.grad_net = res_cnn(18,14)
        
        self._initialize_weights()


        self.r = 1.5
        self.patch_size = (-1,1,128,128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)


    def initialize_ODE_model(self): # Initialise uniquement si nécessaire
        self.ODE_model = NavierStokesFieldEvolution(
                n=self.n, diam=self.diam, T=self.T, max_velocity=self.max_velocity, viscosity=self.viscosity,
                batch_size=self.batch_size, peak_wavenumber=self.peak_wavenumber, scale=self.scale,
                random_state=self.random_state, device = self.device).to(self.device)
        
    def forward(self, x0,mu,y_past,t):

        
        
        if not x0.requires_grad:
            x0.requires_grad = True

        x0 = x0/10.0
        mu = mu/10.0
        
        forecast = torch
        
        cond_input = torch.cat([self.cond_network(mu,t),x0,mu],dim=1)

        
        #norm_x = torch.abs(x0)
        #r = self.r
        #norm_x = torch.max(0*norm_x, norm_x - r)
        
        
        if self.mode == "gradient":
            surface = self.grad_net(cond_input,t)# * torch.exp(-norm_x**2)
            # Calcul du gradient de la surface par rapport à l'entrée x0

            if False:
                grads = []
                for i in range(self.ts):
                    grad_i = torch.autograd.grad(outputs=surface[:,i].sum(), inputs=x0, create_graph=True)[0]
                    grads.append(grad_i)
                grad = torch.cat(grads,dim=1)
            else:
                grad = torch.autograd.grad(outputs=surface.sum(), inputs=x0, create_graph=True)[0]
            #print(grad_i.size(),grad.size(),"grad")
        else:
            grad = self.grad_net(cond_input,t)#[:,1:2] 
            surface = 0
        return grad, surface

    
    
    def ipp_loss(self, score_nn, samples, var_cost, mask,sigma=1, num_samples=1,validation = False):
        """
        Calcule la perte IPP (Information Preserving Prediction) avec l'estimateur de Hutchinson pour la divergence.

        Args:
        - grad_s: le gradient de la sortie par rapport aux échantillons.
        - samples: les échantillons d'entrée.
        - target_values: les valeurs cibles.
        - sigma: paramètre de la distribution normale (par défaut 1).
        - num_samples: nombre de vecteurs de Hutchinson à utiliser pour l'estimation de la divergence (par défaut 1).

        Returns:
        - loss: la perte calculée.
        """
        print("score nn", score_nn.size())
        time_steps = 5
        n = len(samples)
        loss = 0
        
        patch_size_scheduler = [ [n,-1,8,8], [n,-1,8,8], [n,-1,16,16], [n,-1,16,16],[n,-1,16,16], [n,-1,32,32]] #, [n,-1,32,32], [n,-1,32,32]]
        loss = 0

                    # Estimation de la divergence avec Hutchinson
        divergence_s = 0
        grad_norm = 0
        gradient = 0
        sum_mask = 0
        N_h = 1
        for m in range(N_h): # range(mask):
            #m = torch.randn_like(samples).to(self.device)
            
            m = torch.randint_like(samples, low=0, high=2, device=self.device).float()
            m = m * 2 - 1            
            
            '''for i in range(0, 128, 32):
                
                m[i:i+6, :] = 0  # Bandes horizontales
                m[:, i-6:i+6] = 0 '''
            
            sum_mask += m/N_h
            # Calculer le produit des dérivées par r et calculer les gradients
            product_s_m = score_nn * m
            gradient += torch.autograd.grad(
                    outputs=(product_s_m).sum(), 
                    inputs=samples, 
                    create_graph=True
                )[0]*m

            print(f"\033[94mGradient size: {gradient.size(), product_s_m.size()}\033[0m")
    
        # Calcul de la norme du gradient
        sum_mask = sum_mask !=0
        gradient = gradient/N_h
        #square_coeffs = score_nn**2 #(sum_mask*score_nn**2)
        square_coeffs = (score_nn**2)
        ipp_term_mean = 0

        kernel_size = 16
        stride = 8
        for i in range(1):
            #target_values = var_cost[:,2*i:2*i+2].sum(dim=1)[:,None]
            target_values = var_cost.sum(dim=1)[:,None] 

            #print("size target et patch scheduler ",score_nn.size(), target_values.mean(),target_values.size(), patch_size_scheduler[i])
            target_values = (kernel_size**2)*F.avg_pool2d(target_values, kernel_size,stride)
            #target_values = target_values.view(patch_size_scheduler[i])
            #print(target_values.size(),"garget value size")

            #target_values = target_values.sum(dim=(2,3))

            grad_s = score_nn#(sum_mask*score_nn).view(patch_size_scheduler[i])#(sum_mask*score_nn).view(patch_size_scheduler[i])

            
            #samples_reshaped = samples.view(patch_size_scheduler[i])
            # Calcul du gradient du log-p de la distribution normale
            
            if i >-1:
                grad_log_p = -samples/ sigma**2
                product = (grad_s * grad_log_p)

                #print(product.size(),sum_mask.size(),grad_s.size(),grad_log_p.size(),"product size")

                #product = product.view(patch_size_scheduler[i])
                
                ipp_term = (kernel_size**2)*F.avg_pool2d(product,kernel_size ,stride=stride)* target_values#torch.sum(product, dim=(2,3)) * target_values
                
                print(ipp_term.size(), product.size(),target_values.size(), "size ipp term")
                ipp_term_mean += 1*torch.mean(ipp_term)
            
                        
            #grad_r = gradient.view(patch_size_scheduler[i])
            #print(grad_r.size(), r.size())
            # Hutchinson: somme des produits scalaires des gradients par le vecteur aléatoire
            divergence_s = (kernel_size**2)*F.avg_pool2d(gradient,kernel_size,stride=stride)#torch.sum(grad_r, dim=(2,3))


            #print(divergence_s.size(), target_values.size(),ipp_term.size(),grad_s.size())
            divergence_s *= target_values
            ###
            grad_norm = (kernel_size**2)*F.avg_pool2d(square_coeffs,kernel_size,stride=stride)# torch.sum(square_coeffs.view(patch_size_scheduler[i]),dim=(2,3))
            # Calcul final de la perte
            loss += 2 * torch.mean(divergence_s) 
            
        loss += 2*ipp_term_mean + torch.mean(grad_norm)

        return loss



        # Fonction cible
    '''def target_function(self,x, mu, obs):
        norm_x = torch.norm(x, dim=1) / 2
        r = self.r
        norm_x = torch.max(0*norm_x, norm_x - r)
        
        return torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1]) * x[:,2]*x[:,1]* torch.exp(-norm_x**2)'''


    def target_function(self, perturbed_state, state, future_obs, batch_size, ref_forecast = None):
        
        ODE_model = self.ODE_model
        #self.ODE_model.reinstance_grid(self.device)

            
        
        # Prépare les états `perturbed_state` et `state` en utilisant `torch.as_tensor` pour éviter des copies
        perturbed_state = torch.as_tensor(perturbed_state[:, 0], device=self.device)
        state = torch.as_tensor(state[:, 0], device=self.device)
    
        print(perturbed_state.device, ODE_model.device, ODE_model.grid.device,"perturbed state device")
    
        # Utilise `ODE_model` pour faire des prédictions
        forecast = ODE_model(torch.tensor(perturbed_state.detach(), dtype = torch.float64)).to(self.device).detach()
        forecast = torch.tensor(forecast,dtype = torch.float32)
        
        if ref_forecast is None:
            forecast_0 = ODE_model(state.detach()).to(self.device).detach()
        else:
            forecast_0 = ref_forecast.detach()
            
        self.mask = self.mask.to(self.device)
        #print(forecast.size(), forecast_0.size(), perturbed_state.size(), state.size(), "target f")
        print("future obs size", future_obs.size())
        future_obs = ODE_model(torch.as_tensor(future_obs[:,0],device = self.device)).to(self.device)
        
        likelihood_of_forecast = self.mask * ((forecast - future_obs) ** 2)
        likelihood_of_forecast_mu =  self.mask * ((forecast_0 - future_obs) ** 2)
    
        diff = likelihood_of_forecast - likelihood_of_forecast_mu

                # Création d'une copie du tenseur de sortie
        delta_ll = diff.clone()
        
        # Boucle sur les blocs de taille N
        N = batch_size
        split_idx = 20
        if False:
            for i in range(N, delta_ll.size(0), N):
                # Calcul de la différence avec le bloc précédent (i.e., i - N)
                delta_ll[i:i+N] -= diff[i-N:i]
                #delta_ll[i:i+N] -= diff[split_idx*N:split_idx*N + N]
        return delta_ll.float().detach(),diff.detach()

    
    def training_step(self, batch, batch_idx):
        mu,future_obs = batch
        dt =2
        obs = mu
        

        batch_size = len(mu)
        
        # Création d'une image vide (masque initial à 0)
        
        mask = torch.zeros((128, 128), device = self.device)
        mask1 = torch.tensor(mask)
        mask2 = torch.tensor(mask)
        mask3 = torch.tensor(mask)
        mask4 = torch.tensor(mask)

        for i in range(2):
            # Générer des positions de départ aléatoires entre 3 et 5 pour x et y
            x_start = random.randint(6, 10)
            y_start = random.randint(6, 10)
            
            # Utilisation du slicing en démarrant aux positions aléatoires et en plaçant un pixel tous les 8 pixels
            spatial_dep = random.randint(14, 16)
            if i==0:
                mask1[x_start::spatial_dep, y_start::spatial_dep] = 1
            else:
                if i==1:
                    mask2[x_start::spatial_dep, y_start::spatial_dep] = 1
                    mask2 = 0
                    mask2 -= mask2*mask1
                    mask = mask1 + mask2
                if i==2:
                    mask3[x_start::spatial_dep, y_start::spatial_dep] = 1
                    mask3 -= mask*mask3
                    mask += mask3
                if i==3:
                    mask4[x_start::spatial_dep, y_start::spatial_dep] = 1
                    mask4 -= mask*mask4
                    mask += mask4
        #mask = mask1+mask2
        mask = 0*mask1 + 1
        '''for i in range(0, 128, 32):
            mask[i:i+8, :] = 0  # Bandes horizontales
            mask[:, i-8:i+8] = 0  # Bandes verticales'''

        K = 0 #(batch_idx % 3)*self.time_steps
        t = self.time_steps + K
        noise_level = 1e-4
        #if batch_idx %2 == 1:
        #    noise_level *= 8e-1
        step_size = self.step_size
        xk = torch.tensor(mu)
        out_langevin, length_ts = self.langevin_sampling(xk, mu, num_iterations=t, noise=noise_level, step_size=step_size, idx_0 = batch_idx%dt, dT = dt) 
        

        future_obs_list = []
        mu_list = []
        ts_list = []
        
        for i in range(length_ts):
            future_obs_list.append(future_obs)
            mu_list.append(mu)
            ts_list.append(0*mu + batch_idx %dt + i*dt)
        future_obs_list = torch.cat(future_obs_list)
        mu_list =  torch.cat(mu_list)
        ts = torch.cat(ts_list)
            

        sigma = (5e-1)*torch.ones((len(out_langevin),1)).view(-1,1,1,1).to(self.device)

        #sigma[int(len(sigma)/2):] /= 2
        #sigma[3*int(len(sigma)/4):] /=2

        
        mu_forecast = self.ODE_model(torch.as_tensor(mu_list[:, 0], device=self.device)).to(self.device)


        noise = 0*torch.randn(out_langevin.size(), device=self.device)#*(1-mask)
        samples = sigma*torch.randn(out_langevin.size(), device=self.device, requires_grad=True)
        #samples = sigma * (2 * torch.rand(out_langevin.size(), device=self.device, requires_grad=True) - 1)

        state = out_langevin + noise
        perturbed_state = samples + state

        #state_f = self.ODE_model(torch.as_tensor(state[:, 0], device=self.device).detach()).to(self.device)

        #print("check taille tenseurs", perturbed_state.size(),out_langevin.size(),future_obs_list.size(),mu_forecast.size(),length_ts)
        target_values,delta_J =  self.target_function(perturbed_state,state,future_obs_list,ref_forecast = mu_forecast,batch_size = batch_size)
        

        
        predicted_gradient, surface = self(perturbed_state, mu_list,None,ts)
        
        
        kernel_size = 8
        stride = 8
        loss =  torch.mean( ((F.avg_pool2d(target_values.sum(dim=1)[:,None], kernel_size,stride)) - F.avg_pool2d(surface.sum(dim=1)[:,None], kernel_size,stride))**2)
        self.l2_cost_epoch.append(loss.clone().cpu().detach())
        

        
        #self.ipp_loss(predicted_gradient, samples, 0.1*target_values,[mask1,mask2,mask3,mask4],sigma)



        dll,grad_effect = self.target_function(out_langevin[-len(mu):],mu_list[-len(mu):],future_obs_list[-len(mu):],batch_size = batch_size)
        #grad_effect = self.target_function(out_langevin,mu,future_obs)
        
            
        #loss *= 1e-4
        # Calcul de l'erreur entre les gradients réels et prédits
        #grad_effect = self.target_function(out_langevin,mu,future_obs)

        print("\033[31m Degrowth ? \033[0m" ,torch.mean(grad_effect))#, torch.mean((out_langevin - mu)**2))
    
        # Journalisation de la perte et de l'erreur de gradient
        self.log("train_loss", loss)
        torch.save(self.state_dict(),"weights_last_iter_unettanh"+str(self.mode))
        if batch_idx%2 == 0:
            torch.save(delta_J, "deltaj")
        # Utilisation de logging pour afficher les informations
        print(f"Gradient Difference: {loss.item()}")
        print("std pred grad", predicted_gradient.size(),predicted_gradient.sum(dim=1).std(dim=((1,2))).std(),target_values.mean())
        return loss


    
    def validation_step(self, batch, batch_idx):
        
        if self.reinitialize_solver == 1:  
            self.initialize_ODE_model()
            self.reinitalize_solver = 0
            
        with torch.set_grad_enabled(True):  # Activation des gradients
    
            mu,future_obs = batch
    
            
            sigma = 5e-2 
            batch_size = len(mu)
            
            # Création d'une image vide (masque initial à 0)
            
            mask = torch.zeros((128, 128), device = self.device)
            mask1 = torch.tensor(mask)

            K = 0
            t = self.time_steps
            step_size = self.step_size
            
            xk = torch.tensor(mu)
            out_langevin, length_ts = self.langevin_sampling(xk, torch.tensor(mu), num_iterations=self.time_steps, noise=1e-3, step_size=step_size, idx_0 = K, dT = 1,validation = True) 
            
            samples = torch.tensor(out_langevin[-len(mu):])
    
    
            dll,grad_effect = self.target_function(out_langevin[-len(mu):],torch.tensor(mu),future_obs,batch_size = batch_size)
            #grad_effect = self.target_function(out_langevin,mu,future_obs)
            
                
            # Calcul de l'erreur entre les gradients réels et prédits
            #grad_effect = self.target_function(out_langevin,mu,future_obs)
    
            print("\033[31m Degrowth ? \033[0m" ,torch.mean(grad_effect), torch.mean((samples - mu)**2))#, torch.mean((out_langevin - mu)**2))
            
            self.grad_cost_epoch.append(torch.mean(grad_effect).cpu().detach())
            
        
        
            save_tensor_as_png(samples[5,0], "sample_image.png")
            save_tensor_as_png(mu[5,0],"pooled_images.png")
            save_tensor_as_png(future_obs[5,0],"gt_image.png")

            np.save("pooled_tensor",mu.cpu().detach())
            np.save("samples",samples.cpu().detach())
        
            torch.save(torch.tensor(self.grad_cost_epoch),"grad_cost_epoch")
            torch.save(torch.tensor(self.l2_cost_epoch),"l2_cost_epoch")

            # Calcul de l'erreur entre les gradients réels et prédits

            loss = torch.tensor(0)
            # Journalisation de la perte et de l'erreur de gradient
            self.log("validation_loss", loss)

            # Utilisation de logging pour afficher les informations
            print(f"Gradient Difference validation: {loss.item()}")
            loss = torch.mean(grad_effect)
        return loss

        
    def test_step(self, batch, batch_idx):
        mu,future_obs = batch
        obs = mu
        sigma = 0.1
        # Ignore le batch, et génère des échantillons gaussiens directement
        samples = sigma*torch.randn(len(mu), (1,128,128), device=self.device, requires_grad=True)
    
        # Calcul de la fonction cible
        target_values = self.target_function(samples)
    
        # Calcul du gradient réel de la fonction cible
        true_gradients = torch.autograd.grad(outputs=target_values.sum(), inputs=samples, create_graph=True)[0]
    
        # Prédire les gradients avec le modèle
        predicted_gradient, surface = self(torch.tensor(samples,device = self.device), mu,obs)
    
        # Calcul de la perte IPP
        loss = 5e-3*self.ipp_loss(predicted_gradient, samples, target_values)
    
        # Échantillonnage via Langevin Dynamics
        samples = model.langevin_sampling(mu, mu, num_iterations=100, noise=1e-8, step_size=0.05)
        
        # Calcul de l'erreur entre les gradients réels et prédits
        gradient_difference = torch.mean(torch.sqrt(1e-6 + (predicted_gradient - true_gradients) ** 2))
    
        # Journalisation de la perte et de l'erreur de gradient
        self.log("test_loss", loss)
        self.log("test grad diff", gradient_difference)
    
        # Utilisation de logging pour afficher les informations
        print(f"Gradient Difference test: {loss.item()}")
    
        return loss
    
    
    def langevin_sampling(self, x0, mu, num_iterations=15, noise=0.01, step_size=0.01, idx_0=0, dT=1,validation = False):
        """
        Effectue un échantillonnage via Langevin Dynamics et retourne une série temporelle des échantillons.
    
        Args:
        - x0: condition initiale (torch.Tensor).
        - mu: paramètres de conditionnement (torch.Tensor).
        - num_iterations: nombre d'itérations de Langevin Dynamics.
        - noise: intensité du bruit ajouté à chaque étape.
        - step_size: taille des pas de Langevin (contrôle la vitesse de convergence).
        - idx_0: indice initial pour commencer la série temporelle.
        - dT: intervalle entre les échantillons successifs dans la série temporelle.
    
        Returns:
        - samples_series: une série temporelle des échantillons, empilée le long de la première dimension.
        - series_length: longueur de la série temporelle (nombre de points collectés).
        """
        xk = x0.clone().to(self.device).detach().requires_grad_(True)  # Cloner et détacher x0 pour ne pas le modifier
        mu = mu.to(self.device)
        
        # Liste pour stocker les échantillons de la série temporelle
        samples_series = []
        samples_series.append(xk)
        
        # Collecter les échantillons commençant à idx_0 et espacés de dT
        for t in range(num_iterations):

            
            # Calculer le gradient et la surface
            if t == 0:
                grad= 0*mu.clone().detach()
            grad, surface = self(xk, mu, grad.clone().detach(), t)
            
            # Terme de bruit gaussien
            noise_term = noise * torch.randn_like(xk)
            
            # Mise à jour de Langevin
            xk = xk - (step_size / 2) * grad + torch.sqrt(torch.tensor(step_size)) * noise_term
            
            # S'assurer que xk est différentiable sans dépendance des étapes précédentes
            xk = xk.detach().requires_grad_(True)
            
            # Collecter l'échantillon si l'indice correspond à la règle idx_0 + n * dT
            if (t >= idx_0) and ((t - idx_0) % dT == 0):
                samples_series.append(xk)
            
            # Vider le cache pour éviter l'accumulation de mémoire
            torch.cuda.empty_cache()
            step_size *= 1
            noise *= 0.99
            
            noise = max(noise, 5e-3)
        # Stack les échantillons le long de la première dimension

            
        concat_samples = torch.cat(samples_series, dim=0)
        series_length = len(samples_series)

        out = concat_samples
        if validation == True:
            out = xk
        
        return out, series_length
    
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay = 1e-4)

    def _initialize_weights(self, scale_factor=0.1):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialisation des poids pour Conv2d en fonction de l'activation
                if hasattr(m, 'activation'):
                    if isinstance(m.activation, nn.ReLU) or isinstance(m.activation, nn.Softplus):  # ReLU ou SoftReLU
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(m.activation, nn.GELU):  # GELU
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                    elif isinstance(m.activation, nn.Tanh):  # Tanh
                        nn.init.xavier_normal_(m.weight)
                    else:
                        nn.init.xavier_uniform_(m.weight)  # Par défaut pour d'autres activations
                else:
                    # Si aucune activation n'est spécifiée, utiliser ReLU par défaut
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
                # Réduction des poids avec un facteur d'échelle
                m.weight.data *= scale_factor
    
                # Initialisation des biais
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

                    
def save_tensor_as_png(tensor, filename="sample.png"):
    """
    Sauvegarde un tenseur PyTorch sous forme d'image PNG.
    
    Args:
    - tensor: un tenseur PyTorch 2D (H, W) ou 3D (C, H, W) si RGB.
    - filename: nom du fichier de sortie (avec extension .png).
    """
    # Vérifier que le tenseur est en 2D (H, W)
    tensor = tensor.detach().cpu()  # Détacher et amener sur CPU si nécessaire
    
    # Normaliser les valeurs pour qu'elles soient dans la plage [0, 255]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalisation
    tensor = (tensor * 255).to(torch.uint8)  # Conversion en valeurs 8 bits (0-255)
    
    # Convertir le tenseur en numpy array pour PIL
    tensor_np = tensor.numpy()
    
    # Créer une image à partir du tableau numpy
    image = Image.fromarray(tensor_np)
    
    # Sauvegarder en PNG
    image.save(filename)
