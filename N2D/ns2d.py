# -*- coding: utf-8 -*-
import torch
from utils_flows import *

import torch
import torch.fft as fft

from torch_cfd.grids import *
from torch_cfd.equations import *
from torch_cfd.initial_conditions import *
from torch_cfd.finite_differences import *
from torch_cfd.forcings import *

import xarray




class NavierStokesFieldEvolution(nn.Module):
    def __init__(self, n, diam, T, max_velocity, viscosity, batch_size, peak_wavenumber, scale, random_state,device):
        super(NavierStokesFieldEvolution, self).__init__()

        # Paramètres du modèle
        self.n = n
        self.diam = diam
        self.T = T
        dt = 1e-3
        self.num_steps = int(T / dt)
        self.max_velocity = max_velocity
        self.viscosity = viscosity
        self.batch_size = batch_size
        self.peak_wavenumber = peak_wavenumber
        self.scale = scale
        self.random_state = random_state
        # Paramètre factice pour gérer l’appareil automatiquement
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.init_flag = 1

        # Grille
        self.grid = Grid(shape=(n, n), domain=((0, diam), (0, diam)), device=device)

        # Calcul du pas de temps stable
        dx = 2 * torch.pi / n
        dt = 1e-3  # initial dt
        self.dt = stable_time_step(dx=dx, dt=dt, viscosity=viscosity, max_velocity=max_velocity)

        # Nombre d'étapes et d'instantanés
        #self.num_steps = int(T / self.dt)
        
        self.num_snapshots = 10
        self.record_iters = 30

        # Forçage Kolmogorov
        self.forcing_fn = KolmogorovForcing(grid=self.grid, scale=scale, k=peak_wavenumber, vorticity=True)

        # Initialisation du solveur Navier-Stokes
        self.ns2d = NavierStokes2DSpectral(
            viscosity=self.viscosity,
            grid=self.grid,
            drag=0.1,
            smooth=True,
            forcing_fn=self.forcing_fn,
            solver=rk4_crank_nicolson,
        ).to(device)

        # Champ de vorticité initial dans l'espace image
        self.vort_init = torch.stack([
            vorticity_field(self.grid, peak_wavenumber, random_state=random_state + i).data
            for i in range(batch_size)
        ]).to(device)

    @property
    def device(self):
        return self.dummy_param.device
    
    def reinitialize(self,device):
        """Initialise la grille sur le même appareil que `dummy_param`."""
        self.grid = Grid(shape=(self.n, self.n), domain=((0, self.diam), (0, self.diam)), device=device)
        
        self.forcing_fn = KolmogorovForcing(grid=self.grid, scale=self.scale, k=self.peak_wavenumber, vorticity=True)

        # Initialisation du solveur Navier-Stokes
        self.ns2d = NavierStokes2DSpectral(
            viscosity=self.viscosity,
            grid=self.grid,
            drag=0.1,
            smooth=True,
            forcing_fn=self.forcing_fn,
            solver=rk4_crank_nicolson,
        ).to(self.device)

        # Champ de vorticité initial dans l'espace image
        self.vort_init = torch.stack([
            vorticity_field(self.grid, self.peak_wavenumber, random_state=self.random_state + i).data
            for i in range(self.batch_size)
        ]).to(self.device)

    

    def forward(self, vort_init):
        """Simule l'évolution temporelle du champ de vorticité."""
        # S’assurer que la grille est sur le bon appareil avant chaque forward
        if self.init_flag == 1:
            self.reinitialize(self.device)
            self.init_flag = 0

        # Transformation dans l'espace spectral sur le bon appareil
        vort_hat = fft.rfft2(vort_init).to(self.dummy_param.device)
        # Simuler la trajectoire en utilisant le solveur Navier-Stokes avec RK4
        result = get_trajectory_rk4(
            self.ns2d, vort_hat, self.dt,
            num_steps=self.num_steps,
            record_every_steps=self.record_iters,
            pbar=False
        )

        # Retourner les champs de vorticité dans l'espace des images
        vorticity_time_series = fft.irfft2(result["vorticity"])

        return vorticity_time_series

