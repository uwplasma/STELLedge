#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sol_fci_essos.py
================

Flux-Coordinate Independent (FCI) SOL Transport Solver with ESSOS B-field
=========================================================================

This script implements a fully-coupled, flux–coordinate independent (FCI)
scrape-off layer (SOL) transport model in 3D Cartesian coordinates, suitable
as a companion code for a PRL on stellarator edge modeling.

We solve a reduced Braginskii-like system for:
    n(x, t)   : plasma density
    u∥(x, t)  : ion parallel velocity along B
    T_e(x, t) : electron temperature

on a 3D Cartesian grid, using:
  • An anisotropic, field-aligned transport operator based on the
    flux-coordinate independent (FCI) approach of Hariri & Ottaviani:
      - Parallel derivatives along B via field-line following and
        trilinear interpolation.
      - Perpendicular diffusion via standard finite differences, with
        ∇⊥² = ∇² - (b·∇)².
  • A Biot–Savart magnetic field B(x) from ESSOS:
      B(x) = field.B_contravariant(x), in Cartesian coordinates.
  • A plasma–wall mask from a geometric surface (here a cylindrical
    placeholder; in production, use Rmn/Zmn Fourier surfaces).
  • Time integration via Diffrax (JAX-based ODE integrator), with
    everything jittable and differentiable.

The model equations (schematic)
-------------------------------

Let b = B/|B| be the unit vector along B. On each grid point we evolve:

(1) Continuity:
    ∂n/∂t = - ∇∥(n u∥) + D⊥ ∇⊥² n + S_n

(2) Parallel momentum (ion fluid):
    ∂u∥/∂t = - u∥ ∂∥ u∥ - (1/m_i) (1/n) ∂∥(n T_e)
              - ν_∥ u∥ + μ⊥ ∇⊥² u∥

(3) Electron temperature (advection + diffusion):
    ∂T_e/∂t = - u∥ ∂∥ T_e - (γ - 1) T_e ∂∥ u∥
              + χ∥ ∂∥² T_e + χ⊥ ∇⊥² T_e + S_T

where:
  • ∂∥ ≡ b·∇ is the parallel derivative,
  • ∂∥² is implemented by FCI using field-line tracing via ±Δs b,
  • perpendicular Laplacian uses standard finite differences.

Boundary conditions:
--------------------
  • Plasma domain inside a wall mask: mask=1 inside plasma, 0 outside.
  • Outside mask: (n, u∥, T_e) fixed to (n_wall, 0, T_wall).
  • Along-field boundary at plates is approximated by the wall mask.

This file is self-contained except for:
  • ESSOS (for B field): github.com/uwplasma/ESSOS
  • Diffrax, JAX, Matplotlib

Author: (your name / affiliation)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from jax import debug as jdbg
from jax import tree_util

import diffrax as dfx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from functools import partial

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Array = jnp.ndarray


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SOLConfig:
    """
    Configuration for the SOL FCI solver.

    Grid & domain
    -------------
    Nx, Ny, Nz : int
        Number of grid points in x, y, z.
    xlim, ylim, zlim : (float, float)
        Domain limits in each direction.

    Transport coefficients
    ----------------------
    D_perp : float
        Perpendicular particle diffusion coefficient.
    chi_par : float
        Parallel thermal diffusivity (electrons).
    chi_perp : float
        Perpendicular thermal diffusivity.
    mu_perp : float
        Perpendicular viscosity for u∥.
    nu_par : float
        Parallel friction/damping coefficient for u∥.
    gamma_e : float
        Effective adiabatic index for electrons in parallel compression term.

    Plasma / wall parameters
    ------------------------
    n_wall : float
        Density in wall region (Dirichlet).
    T_wall : float
        Temperature in wall region (Dirichlet).
    m_i : float
        Ion mass (normalized units OK).

    FCI parameters
    --------------
    ds_par : float
        Parallel step length for FCI stencil, in physical units.

    Time integration
    ----------------
    t0, t1 : float
        Start and end times.
    save_every : float
        Time interval between saved snapshots.
    rtol, atol : float
        Tolerances for Diffrax integrator.
    """
    Nx: int = 32
    Ny: int = 32
    Nz: int = 32
    xlim: Tuple[float, float] = (-1.5, 1.5)
    ylim: Tuple[float, float] = (-1.5, 1.5)
    zlim: Tuple[float, float] = (-1.5, 1.5)

    D_perp: float = 5e-3
    chi_par: float = 1.0
    chi_perp: float = 5e-3
    mu_perp: float = 5e-3
    nu_par: float = 0.5
    gamma_e: float = 5.0 / 3.0

    n_wall: float = 1e-4
    T_wall: float = 0.03
    m_i: float = 1.0

    ds_par: float = 0.05

    t0: float = 0.0
    t1: float = 0.05
    save_every: float = 5e-3
    rtol: float = 1e-5
    atol: float = 1e-7


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def build_cartesian_grid(cfg: SOLConfig) -> Tuple[Array, Array, Array, Array]:
    """
    Build 3D Cartesian grid and coordinate array.

    Returns
    -------
    x, y, z : 1D arrays
    xyz : array (Nx, Ny, Nz, 3)
    """
    x = jnp.linspace(cfg.xlim[0], cfg.xlim[1], cfg.Nx)
    y = jnp.linspace(cfg.ylim[0], cfg.ylim[1], cfg.Ny) if False else jnp.linspace(cfg.ylim[0], cfg.ylim[1], cfg.Ny)
    z = jnp.linspace(cfg.zlim[0], cfg.zlim[1], cfg.Nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    xyz = jnp.stack([X, Y, Z], axis=-1)
    return x, y, z, xyz


# ---------------------------------------------------------------------------
# Example wall mask: cylindrical surface
# In production, replace with Rmn/Zmn-based Fourier surface.
# ---------------------------------------------------------------------------

def example_wall_mask(xyz: Array, R_wall: float = 1.2) -> Array:
    """
    Example plasma mask from a cylindrical wall:

      mask = 1 inside plasma (R < R_wall),
             0 outside (wall region).

    Parameters
    ----------
    xyz : array (..., 3)

    Returns
    -------
    mask : array (...,)
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    R = jnp.sqrt(x**2 + y**2)
    mask = (R < R_wall).astype(jnp.float64)
    return mask


# ---------------------------------------------------------------------------
# Trilinear interpolation (for FCI)
# ---------------------------------------------------------------------------

def _trilinear_interp(F: Array,
                      x_grid: Array,
                      y_grid: Array,
                      z_grid: Array,
                      pts: Array) -> Array:
    """
    Trilinear interpolation on a 3D Cartesian grid.

    Parameters
    ----------
    F : array (Nx, Ny, Nz)
        Field to interpolate.
    x_grid, y_grid, z_grid : 1D arrays
    pts : array (N, 3)
        Target points in physical coordinates.

    Returns
    -------
    vals : array (N,)
        Interpolated values F(x, y, z).
    """
    Nx, Ny, Nz = F.shape
    x0, y0, z0 = x_grid[0], y_grid[0], z_grid[0]
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]

    # Clip points to domain interior to avoid out-of-bounds
    epsx = 0.5 * dx
    epsy = 0.5 * dy
    epsz = 0.5 * dz
    x = jnp.clip(pts[:, 0], x_grid[0] + epsx, x_grid[-1] - epsx)
    y = jnp.clip(pts[:, 1], y_grid[0] + epsy, y_grid[-1] - epsy)
    z = jnp.clip(pts[:, 2], z_grid[0] + epsz, z_grid[-1] - epsz)

    ix_f = (x - x0) / dx
    iy_f = (y - y0) / dy
    iz_f = (z - z0) / dz

    ix0 = jnp.floor(ix_f).astype(jnp.int32)
    iy0 = jnp.floor(iy_f).astype(jnp.int32)
    iz0 = jnp.floor(iz_f).astype(jnp.int32)

    ix0 = jnp.clip(ix0, 0, Nx - 2)
    iy0 = jnp.clip(iy0, 0, Ny - 2)
    iz0 = jnp.clip(iz0, 0, Nz - 2)

    tx = ix_f - ix0
    ty = iy_f - iy0
    tz = iz_f - iz0

    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    # Gather corner values
    f000 = F[ix0, iy0, iz0]
    f001 = F[ix0, iy0, iz1]
    f010 = F[ix0, iy1, iz0]
    f011 = F[ix0, iy1, iz1]
    f100 = F[ix1, iy0, iz0]
    f101 = F[ix1, iy0, iz1]
    f110 = F[ix1, iy1, iz0]
    f111 = F[ix1, iy1, iz1]

    # Trilinear interpolation
    c00 = f000 * (1 - tx) + f100 * tx
    c01 = f001 * (1 - tx) + f101 * tx
    c10 = f010 * (1 - tx) + f110 * tx
    c11 = f011 * (1 - tx) + f111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    vals = c0 * (1 - tz) + c1 * tz
    return vals


# Vectorized wrapper for convenience
def trilinear_interp(F: Array,
                     x_grid: Array,
                     y_grid: Array,
                     z_grid: Array,
                     pts: Array) -> Array:
    """
    Vectorized trilinear interpolation.

    Parameters
    ----------
    F : array (Nx, Ny, Nz)
    x_grid, y_grid, z_grid : 1D arrays
    pts : array (N, 3)

    Returns
    -------
    vals : array (N,)
    """
    return _trilinear_interp(F, x_grid, y_grid, z_grid, pts)


# ---------------------------------------------------------------------------
# FCI parallel derivatives
# ---------------------------------------------------------------------------

def fci_parallel_derivs(field: Array,
                        xyz: Array,
                        b_grid: Array,
                        x_grid: Array,
                        y_grid: Array,
                        z_grid: Array,
                        ds: float) -> Tuple[Array, Array]:
    """
    Compute FCI-based parallel derivatives ∂∥f and ∂∥² f on a 3D grid.

    Using Hariri-style field-line parallelogram:
      f±(x) = f(x ± ds b)
      ∂∥ f ≈ (f+ - f-) / (2 ds)
      ∂∥² f ≈ (f+ - 2 f0 + f-) / ds²

    Parameters
    ----------
    field : array (Nx, Ny, Nz)
    xyz : array (Nx, Ny, Nz, 3)
    b_grid : array (Nx, Ny, Nz, 3)
    x_grid, y_grid, z_grid : 1D arrays
    ds : float
        Parallel step length.

    Returns
    -------
    dpar_f : array (Nx, Ny, Nz)
        First parallel derivative ∂∥ f.
    d2par_f : array (Nx, Ny, Nz)
        Second parallel derivative ∂∥² f.
    """
    Nx, Ny, Nz = field.shape
    N = Nx * Ny * Nz

    pts = xyz.reshape((N, 3))
    b_flat = b_grid.reshape((N, 3))
    f0 = field.reshape((N,))

    pts_plus = pts + ds * b_flat
    pts_minus = pts - ds * b_flat

    f_plus = trilinear_interp(field, x_grid, y_grid, z_grid, pts_plus)
    f_minus = trilinear_interp(field, x_grid, y_grid, z_grid, pts_minus)

    dpar = (f_plus - f_minus) / (2.0 * ds)
    d2par = (f_plus - 2.0 * f0 + f_minus) / (ds**2)

    return dpar.reshape((Nx, Ny, Nz)), d2par.reshape((Nx, Ny, Nz))


# ---------------------------------------------------------------------------
# Perpendicular Laplacian via finite differences
# ---------------------------------------------------------------------------

def second_derivatives_cartesian(F: Array,
                                 dx: float,
                                 dy: float,
                                 dz: float) -> Tuple[Array, Array, Array, Array]:
    """
    Compute Cartesian second derivatives and Laplacian with periodic BCs.

    Parameters
    ----------
    F : array (Nx, Ny, Nz)
    dx, dy, dz : float

    Returns
    -------
    F_xx, F_yy, F_zz, lap
    """
    F_xx = (jnp.roll(F, -1, axis=0) - 2.0 * F + jnp.roll(F, 1, axis=0)) / dx**2
    F_yy = (jnp.roll(F, -1, axis=1) - 2.0 * F + jnp.roll(F, 1, axis=1)) / dy**2
    F_zz = (jnp.roll(F, -1, axis=2) - 2.0 * F + jnp.roll(F, 1, axis=2)) / dz**2
    lap = F_xx + F_yy + F_zz
    return F_xx, F_yy, F_zz, lap


def perp_laplacian(F: Array,
                   xyz: Array,
                   b_grid: Array,
                   x_grid: Array,
                   y_grid: Array,
                   z_grid: Array,
                   ds: float,
                   dx: float,
                   dy: float,
                   dz: float) -> Tuple[Array, Array, Array]:
    """
    Compute ∇⊥² F using:

        ∇⊥² F = ∇² F - ∂∥² F

    Returns
    -------
    lap_perp : array (Nx, Ny, Nz)
    dpar_f : array (Nx, Ny, Nz)
    d2par_f : array (Nx, Ny, Nz)
    """
    F_xx, F_yy, F_zz, lap = second_derivatives_cartesian(F, dx, dy, dz)
    dpar_f, d2par_f = fci_parallel_derivs(F, xyz, b_grid, x_grid, y_grid, z_grid, ds)
    lap_perp = lap - d2par_f
    return lap_perp, dpar_f, d2par_f


# ---------------------------------------------------------------------------
# PDE parameters container
# ---------------------------------------------------------------------------

@tree_util.register_pytree_node_class
@dataclass
class SOLParams:
    cfg: SOLConfig
    x_grid: Array
    y_grid: Array
    z_grid: Array
    xyz: Array              # (Nx, Ny, Nz, 3)
    b_grid: Array           # (Nx, Ny, Nz, 3)
    mask_plasma: Array      # (Nx, Ny, Nz)
    dx: float
    dy: float
    dz: float
    S_n: Array              # (Nx, Ny, Nz) density source
    S_T: Array              # (Nx, Ny, Nz) temperature source

    # --- NEW: PyTree API for SOLParams ---

    def tree_flatten(self):
        """
        Children = JAX arrays (dynamic leaves),
        aux_data = small static stuff (cfg and spacings).
        """
        children = (
            self.x_grid,
            self.y_grid,
            self.z_grid,
            self.xyz,
            self.b_grid,
            self.mask_plasma,
            self.S_n,
            self.S_T,
        )
        aux_data = (self.cfg, self.dx, self.dy, self.dz)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        cfg, dx, dy, dz = aux_data
        (
            x_grid,
            y_grid,
            z_grid,
            xyz,
            b_grid,
            mask_plasma,
            S_n,
            S_T,
        ) = children
        return cls(
            cfg=cfg,
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=z_grid,
            xyz=xyz,
            b_grid=b_grid,
            mask_plasma=mask_plasma,
            dx=dx,
            dy=dy,
            dz=dz,
            S_n=S_n,
            S_T=S_T,
        )


# ---------------------------------------------------------------------------
# Example sources
# ---------------------------------------------------------------------------

def make_sources(xyz: Array) -> Tuple[Array, Array]:
    """
    Example sources S_n and S_T.

    For demonstration, we use a Gaussian heat source and a small particle
    source localized near outboard midplane.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    R = jnp.sqrt(x**2 + y**2)
    R0 = 0.9
    z0 = 0.0
    width_R = 0.2
    width_z = 0.3

    # Particle source
    S_n0 = 1.0
    S_n = S_n0 * jnp.exp(-((R - R0)**2 / (2 * width_R**2) +
                           (z - z0)**2 / (2 * width_z**2)))

    # Heat source
    S_T0 = 10.0
    S_T = S_T0 * jnp.exp(-((R - R0)**2 / (2 * width_R**2) +
                           (z - z0)**2 / (2 * width_z**2)))

    return S_n, S_T


# ---------------------------------------------------------------------------
# RHS of coupled SOL system
# ---------------------------------------------------------------------------

# @jit
def rhs_sol(t: float, y_flat: Array, params: SOLParams) -> Array:
    """
    RHS of the coupled SOL equations:

        ∂n/∂t   = - ∂∥ (n u)    + D⊥ ∇⊥² n + S_n
        ∂u/∂t   = - u ∂∥ u
                  - (1/m_i)(1/n) ∂∥ (n T)
                  - ν∥ u
                  + μ⊥ ∇⊥² u
        ∂T/∂t   = - u ∂∥ T
                  - (γ_e - 1) T ∂∥ u
                  + χ∥ ∂∥² T
                  + χ⊥ ∇⊥² T
                  + S_T / (3/2 n)

    with Dirichlet-like wall conditions enforced via mask.

    y_flat contains [n_flat, u_flat, T_flat] concatenated.
    """
    cfg = params.cfg
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    N = Nx * Ny * Nz

    n_flat = y_flat[0:N]
    u_flat = y_flat[N:2*N]
    T_flat = y_flat[2*N:3*N]

    n = n_flat.reshape((Nx, Ny, Nz))
    u = u_flat.reshape((Nx, Ny, Nz))
    T = T_flat.reshape((Nx, Ny, Nz))

    mask = params.mask_plasma
    xyz = params.xyz
    b_grid = params.b_grid
    x_grid = params.x_grid
    y_grid = params.y_grid
    z_grid = params.z_grid
    dx, dy, dz = params.dx, params.dy, params.dz

    # Apply wall Dirichlet conditions
    n = mask * n + (1.0 - mask) * cfg.n_wall
    T = mask * T + (1.0 - mask) * cfg.T_wall
    u = mask * u  # u_wall = 0

    # Debug print basic field ranges
    jdbg.print(
        "[rhs] t={t:.3e}  n:[{nmin:.3e},{nmax:.3e}]  "
        "T:[{Tmin:.3e},{Tmax:.3e}]  u:[{umin:.3e},{umax:.3e}]",
        t=t,
        nmin=jnp.min(n),
        nmax=jnp.max(n),
        Tmin=jnp.min(T),
        Tmax=jnp.max(T),
        umin=jnp.min(u),
        umax=jnp.max(u),
    )

    # ----------------------------------------------------------------------
    # Continuity: ∂n/∂t = - ∂∥(n u) + D⊥ ∇⊥² n + S_n
    # ----------------------------------------------------------------------
    n_u = n * u
    lap_perp_n, dpar_n, d2par_n = perp_laplacian(
        n, xyz, b_grid, x_grid, y_grid, z_grid,
        cfg.ds_par, dx, dy, dz
    )
    # Parallel derivative of flux n u
    dpar_nu, _ = fci_parallel_derivs(
        n_u, xyz, b_grid, x_grid, y_grid, z_grid, cfg.ds_par
    )

    dn_dt = -dpar_nu + cfg.D_perp * lap_perp_n + params.S_n
    dn_dt = mask * dn_dt  # no evolution in wall

    # ----------------------------------------------------------------------
    # Parallel momentum: u = u∥
    # ∂u/∂t = - u ∂∥ u
    #        - (1/m_i)(1/n) ∂∥ (n T)
    #        - ν∥ u + μ⊥ ∇⊥² u
    # ----------------------------------------------------------------------
    lap_perp_u, dpar_u, _ = perp_laplacian(
        u, xyz, b_grid, x_grid, y_grid, z_grid,
        cfg.ds_par, dx, dy, dz
    )
    p = n * T
    dpar_p, _ = fci_parallel_derivs(
        p, xyz, b_grid, x_grid, y_grid, z_grid, cfg.ds_par
    )

    # Avoid division by zero
    n_safe = jnp.maximum(n, 1e-8)

    du_dt = (
        - u * dpar_u
        - (1.0 / cfg.m_i) * (1.0 / n_safe) * dpar_p
        - cfg.nu_par * u
        + cfg.mu_perp * lap_perp_u
    )
    du_dt = mask * du_dt  # no evolve in wall

    # ----------------------------------------------------------------------
    # Electron temperature:
    # ∂T/∂t = - u ∂∥ T
    #        - (γ_e - 1) T ∂∥ u
    #        + χ∥ ∂∥² T
    #        + χ⊥ ∇⊥² T
    #        + S_T / (3/2 n)
    # ----------------------------------------------------------------------
    lap_perp_T, dpar_T, d2par_T = perp_laplacian(
        T, xyz, b_grid, x_grid, y_grid, z_grid,
        cfg.ds_par, dx, dy, dz
    )

    chi_par = cfg.chi_par
    chi_perp = cfg.chi_perp

    dT_dt = (
        - u * dpar_T
        - (cfg.gamma_e - 1.0) * T * dpar_u
        + chi_par * d2par_T
        + chi_perp * lap_perp_T
        + params.S_T / (1.5 * n_safe)
    )
    dT_dt = mask * dT_dt

    # Assemble flattened derivative
    dy_dt = jnp.concatenate(
        [dn_dt.reshape(-1), du_dt.reshape(-1), dT_dt.reshape(-1)],
        axis=0
    )

    # Additional debug: report average parallel derivatives magnitudes
    jdbg.print(
        "[rhs] <|∂∥T|>={dT:.3e}, <|∂∥u|>={du:.3e}",
        dT=jnp.mean(jnp.abs(dpar_T)),
        du=jnp.mean(jnp.abs(dpar_u)),
    )

    return dy_dt


# ---------------------------------------------------------------------------
# Main driver: build params, run Diffrax, and make figures
# ---------------------------------------------------------------------------

def run_sol_fci_demo(cfg: SOLConfig,
                     B_func: Callable[[Array], Array],
                     wall_mask_func: Callable[[Array], Array] = example_wall_mask
                     ) -> Tuple[Array, Array]:
    """
    Run SOL FCI simulation with given config and magnetic field.

    Parameters
    ----------
    cfg : SOLConfig
        Configuration parameters.
    B_func : callable
        B_func(xyz_flat) -> B_flat of shape (N, 3), JAX-compatible.
    wall_mask_func : callable
        wall_mask_func(xyz) -> mask (Nx, Ny, Nz).

    Returns
    -------
    times : array (Nt,)
    y_hist : array (Nt, 3, Nx, Ny, Nz)
        Time history of [n, u, T].
    """
    print("=== SOL FCI ESSOS-based Solver Demo ===")
    print(cfg)

    # Build grid
    x, y, z, xyz = build_cartesian_grid(cfg)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    print(f"[setup] Grid: Nx={cfg.Nx}, Ny={cfg.Ny}, Nz={cfg.Nz}")
    print(f"[setup] dx={dx:.3e}, dy={dy:.3e}, dz={dz:.3e}")

    # Magnetic field from ESSOS (or other Biot–Savart) on grid
    xyz_flat = xyz.reshape((-1, 3))
    B_flat = B_func(xyz_flat)          # (N, 3)
    B_grid = B_flat.reshape((cfg.Nx, cfg.Ny, cfg.Nz, 3))

    B_mag = jnp.sqrt(jnp.sum(B_grid**2, axis=-1) + 1e-16)
    b_grid = B_grid / B_mag[..., None]

    print("[setup] B field computed on grid.")
    print(f"[setup] <|B|> = {float(jnp.mean(B_mag)):.3e}, "
          f"|B|_min = {float(jnp.min(B_mag)):.3e}, "
          f"|B|_max = {float(jnp.max(B_mag)):.3e}")

    # Wall / plasma mask
    mask_plasma = wall_mask_func(xyz)
    print("[setup] Plasma mask constructed from wall.")
    print(f"[setup] Plasma volume fraction: {float(jnp.mean(mask_plasma)):.3f}")

    # Sources
    S_n, S_T = make_sources(xyz)
    print("[setup] Sources S_n and S_T initialized.")

    # Initial conditions: near-uniform n, small u, localized T bump
    n0 = cfg.n_wall + mask_plasma * 0.9
    u0 = jnp.zeros_like(n0)
    R = jnp.sqrt(xyz[..., 0]**2 + xyz[..., 1]**2)
    T0 = cfg.T_wall + mask_plasma * 0.5 * jnp.exp(-((R - 0.8)**2 / (2 * 0.1**2)))

    print(f"[IC] n0:[{float(jnp.min(n0)):.3e},{float(jnp.max(n0)):.3e}] "
          f"T0:[{float(jnp.min(T0)):.3e},{float(jnp.max(T0)):.3e}]")

    N = cfg.Nx * cfg.Ny * cfg.Nz
    y0 = jnp.concatenate([n0.reshape(-1), u0.reshape(-1), T0.reshape(-1)], axis=0)

    params = SOLParams(
        cfg=cfg,
        x_grid=x,
        y_grid=y,
        z_grid=z,
        xyz=xyz,
        b_grid=b_grid,
        mask_plasma=mask_plasma,
        dx=dx,
        dy=dy,
        dz=dz,
        S_n=S_n,
        S_T=S_T,
    )

    term = dfx.ODETerm(rhs_sol)
    save_ts = jnp.arange(cfg.t0, cfg.t1 + 1e-12, cfg.save_every)
    saveat = dfx.SaveAt(ts=save_ts, t0=True, t1=True)

    # Explicit solver to avoid optimistix/lineax implicit machinery
    solver = dfx.Tsit5()

    # You can keep the PID step-size controller; Tsit5 supports it
    stepsize_controller = dfx.PIDController(rtol=cfg.rtol, atol=cfg.atol)

    print("[time] Starting Diffrax solve (Tsit5 explicit)...")
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=cfg.t0,
        t1=cfg.t1,
        dt0=1e-5,              # a bit smaller initial dt than before
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=1_000_000,   # give it room; explicit steps will be smaller
    )
    print("[time] Diffrax solve complete.")

    times = sol.ts
    ys = sol.ys  # shape (Nt, 3*N)

    Nt = ys.shape[0]
    y_hist = ys.reshape((Nt, 3, cfg.Nx, cfg.Ny, cfg.Nz))

    # Final diagnostics
    n_final = y_hist[-1, 0]
    u_final = y_hist[-1, 1]
    T_final = y_hist[-1, 2]
    print(f"[final] n:[{float(jnp.min(n_final)):.3e},{float(jnp.max(n_final)):.3e}], "
          f"T:[{float(jnp.min(T_final)):.3e},{float(jnp.max(T_final)):.3e}], "
          f"u:[{float(jnp.min(u_final)):.3e},{float(jnp.max(u_final)):.3e}]")

    return np.array(times), np.array(y_hist)


# ---------------------------------------------------------------------------
# Plotting: publication-ready figures
# ---------------------------------------------------------------------------

def make_publication_figures(times: np.ndarray,
                             y_hist: np.ndarray,
                             x: np.ndarray,
                             y: np.ndarray,
                             z: np.ndarray,
                             outdir: str = "figures") -> None:
    """
    Create publication-ready figures from the simulation.

    Figures:
    --------
    1) 2D slice of T_e at midplane z ~ 0, at final time.
    2) 2D slice of n at midplane z ~ 0, at final time.
    3) Radial profile of T_e and n along major radius at z=0, y>0.
    """
    os.makedirs(outdir, exist_ok=True)

    Nt, _, Nx, Ny, Nz = y_hist.shape
    n = y_hist[-1, 0]
    u = y_hist[-1, 1]
    T = y_hist[-1, 2]

    # Find closest z=0 slice
    iz0 = int(np.argmin(np.abs(z)))
    z0 = float(z[iz0])

    # 2D slices at z ~ 0
    n_slice = n[:, :, iz0]
    T_slice = T[:, :, iz0]

    X, Y = np.meshgrid(x, y, indexing="ij")

    # Figure 1: T_e slice
    fig1, ax1 = plt.subplots(figsize=(5.0, 4.0))
    c1 = ax1.pcolormesh(X, Y, T_slice.T, shading="auto")
    fig1.colorbar(c1, ax=ax1, label=r"$T_e$")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_title(rf"$T_e(x,y,z\approx {z0:.2f})$ at $t={times[-1]:.3e}$")
    fig1.tight_layout()
    fig1.savefig(os.path.join(outdir, "Te_slice_midplane.png"), dpi=300)
    plt.close(fig1)

    # Figure 2: n slice
    fig2, ax2 = plt.subplots(figsize=(5.0, 4.0))
    c2 = ax2.pcolormesh(X, Y, n_slice.T, shading="auto")
    fig2.colorbar(c2, ax=ax2, label=r"$n$")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")
    ax2.set_title(rf"$n(x,y,z\approx {z0:.2f})$ at $t={times[-1]:.3e}$")
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "n_slice_midplane.png"), dpi=300)
    plt.close(fig2)

    # Figure 3: radial profile along y>0 line at midplane
    iy_pos = np.argmax(y > 0.0) if np.any(y > 0.0) else Ny // 2
    n_prof = n[:, iy_pos, iz0]
    T_prof = T[:, iy_pos, iz0]
    R_prof = np.sqrt(x**2 + y[iy_pos]**2)

    fig3, ax3 = plt.subplots(figsize=(5.0, 4.0))
    ax3.plot(R_prof, T_prof, label=r"$T_e$")
    ax3.plot(R_prof, n_prof, label=r"$n$")
    ax3.set_xlabel(r"$R$")
    ax3.set_ylabel(r"Profiles (arb. units)")
    ax3.set_title(rf"Radial profiles at $(y={y[iy_pos]:.2f}, z={z0:.2f})$")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(outdir, "radial_profiles.png"), dpi=300)
    plt.close(fig3)

    print(f"[fig] Saved figures to '{outdir}'.")


# ---------------------------------------------------------------------------
# ESSOS wrapper
# ---------------------------------------------------------------------------

def make_essos_B_func(json_file: str):
    """
    Build a JAX-compatible B_func(xyz_flat) using ESSOS BiotSavart.

    Parameters
    ----------
    json_file : str
        Path to ESSOS coil JSON file.

    Returns
    -------
    B_func : callable
        B_func(xyz_flat: (N,3)) -> (N,3).
    """
    from essos.fields import BiotSavart
    from essos.coils import Coils_from_json

    coils = Coils_from_json(json_file)
    field = BiotSavart(coils)

    # ESSOS BiotSavart.B_contravariant expects *one* point of shape (3,)
    # We vmap over the first axis of xyz_flat, which has shape (N, 3)
    @jit
    def B_func(xyz_flat: Array) -> Array:
        # xyz_flat: (N, 3)
        return vmap(field.B_contravariant)(xyz_flat)  # -> (N, 3)

    return B_func, coils

def trace_field_line_with_connection(
    x0: np.ndarray,
    B_func: Callable[[Array], Array],
    wall_mask_func: Callable[[Array], Array],
    cfg: SOLConfig,
    ds: float,
    max_steps: int,
) -> Tuple[np.ndarray, float]:
    """
    Trace a single field line starting at x0, stepping along +b with step ds,
    until it hits the wall (mask=0) or leaves the domain.

    Returns
    -------
    pts : array (n_pts, 3)
        Coordinates along the field line.
    Lpar : float
        Connection length along the traced segment (≈ (n_pts-1)*ds).
    """
    pts = np.zeros((max_steps, 3))
    pts[0] = x0

    for k in range(max_steps - 1):
        xk = jnp.asarray(pts[k])[None, :]      # (1,3)
        Bk = B_func(xk)                        # (1,3)
        Bk_np = np.array(Bk)[0]
        Bmag = np.linalg.norm(Bk_np)
        if Bmag < 1e-10:
            n_pts = k + 1
            Lpar = ds * (n_pts - 1)
            return pts[:n_pts], Lpar

        b = Bk_np / Bmag
        x_next = pts[k] + ds * b

        # Domain bounds
        if not (cfg.xlim[0] <= x_next[0] <= cfg.xlim[1] and
                cfg.ylim[0] <= x_next[1] <= cfg.ylim[1] and
                cfg.zlim[0] <= x_next[2] <= cfg.zlim[1]):
            n_pts = k + 1
            Lpar = ds * (n_pts - 1)
            return pts[:n_pts], Lpar

        # Wall / plasma mask check at the new point
        x_next_jax = jnp.asarray(x_next)[None, None, None, :]  # (1,1,1,3)
        mask_val = np.array(wall_mask_func(x_next_jax))[0, 0, 0]
        if mask_val < 0.5:
            pts[k+1] = x_next
            n_pts = k + 2
            Lpar = ds * (n_pts - 1)
            return pts[:n_pts], Lpar

        pts[k+1] = x_next

    n_pts = max_steps
    Lpar = ds * (n_pts - 1)
    return pts, Lpar


def surface_points_from_mask(xyz: Array, mask_plasma: Array) -> np.ndarray:
    """
    Extract approximate surface points from a binary mask by selecting
    voxels that are plasma but have at least one non-plasma neighbor.
    """
    mask = np.array(mask_plasma)
    xyz_np = np.array(xyz)

    # pad mask to handle edges
    pad = np.pad(mask, 1, mode="edge")
    surf = np.zeros_like(mask, dtype=bool)

    # 6-neighbor stencil
    for axis in range(3):
        for shift in (-1, 1):
            shifted = np.roll(pad, shift=shift, axis=axis+0)[1:-1, 1:-1, 1:-1]
            surf |= (mask == 1.0) & (shifted == 0.0)

    pts = xyz_np[surf]
    return pts

def cylinder_surface_points(R_wall: float,
                            zlim: Tuple[float, float],
                            n_theta: int = 128,
                            n_z: int = 64) -> np.ndarray:
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    z = np.linspace(zlim[0], zlim[1], n_z)
    Theta, Z = np.meshgrid(theta, z, indexing="ij")
    X = R_wall * np.cos(Theta)
    Y = R_wall * np.sin(Theta)
    pts = np.stack([X, Y, Z], axis=-1)  # (n_theta, n_z, 3)
    return pts.reshape(-1, 3)

def make_3d_geometry_figure(
    cfg: SOLConfig,
    coils,
    B_func: Callable[[Array], Array],
    wall_mask_func: Callable[[Array], Array],
    R0: float,
    a: float,
    n_fieldlines: int = 24,
    fieldline_length: float = 4.0,
    fieldline_steps: int = 400,
    use_voxel_surface: bool = False,
    outdir: str = "figures",
) -> None:
    """
    PRL-style geometry panel for a circular torus:
      left: coils + toroidal SOL surface + field lines colored by L_parallel
      right: connection length vs poloidal angle theta.
    """
    os.makedirs(outdir, exist_ok=True)

    # Rebuild grid + mask (for optional voxel surface)
    x, y, z, xyz = build_cartesian_grid(cfg)
    mask_plasma = wall_mask_func(xyz)
    mask_np = np.array(mask_plasma)

    # ------------------------------------------------------------------
    # Field-line starting points: scan poloidal angle theta at fixed
    # toroidal angle zeta = 0 on a ring just inside the torus.
    # ------------------------------------------------------------------
    theta_pol = np.linspace(0.0, 2.0 * np.pi, n_fieldlines, endpoint=False)
    zeta0 = 0.0
    r_minor = 0.9 * a   # slightly inside the SOL boundary

    X0 = (R0 + r_minor * np.cos(theta_pol)) * np.cos(zeta0)
    Y0 = (R0 + r_minor * np.cos(theta_pol)) * np.sin(zeta0)
    Z0 = r_minor * np.sin(theta_pol)
    starts = np.stack([X0, Y0, Z0], axis=1)

    # Keep only starts actually inside plasma mask
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])

    def nearest_idx(val, grid, d):
        return np.clip(
            np.round((val - grid[0]) / d).astype(int),
            0,
            len(grid) - 1,
        )

    ix = nearest_idx(starts[:, 0], np.array(x), dx)
    iy = nearest_idx(starts[:, 1], np.array(y), dy)
    iz = nearest_idx(starts[:, 2], np.array(z), dz)

    inside = mask_np[ix, iy, iz] > 0.5
    starts = starts[inside]
    theta_pol = theta_pol[inside]
    if starts.shape[0] == 0:
        raise RuntimeError(
            "No valid start points for field lines on the poloidal ring; "
            "check R0, a, or mask."
        )

    # Step size along field line
    ds = fieldline_length / fieldline_steps

    # ------------------------------------------------------------------
    # Smooth circular torus surface: (theta_pol, zeta_tor)
    # ------------------------------------------------------------------
    n_theta_surf = 96
    n_zeta_surf = 128
    theta_surf = np.linspace(0.0, 2.0 * np.pi, n_theta_surf, endpoint=False)
    zeta_surf = np.linspace(0.0, 2.0 * np.pi, n_zeta_surf, endpoint=False)
    Theta_s, Zeta_s = np.meshgrid(theta_surf, zeta_surf, indexing="ij")

    X_s = (R0 + a * np.cos(Theta_s)) * np.cos(Zeta_s)
    Y_s = (R0 + a * np.cos(Theta_s)) * np.sin(Zeta_s)
    Z_s = a * np.sin(Theta_s)

    # Optional voxel-based "cloud" of plasma boundary points
    if use_voxel_surface:
        voxel_pts = surface_points_from_mask(xyz, mask_plasma)
    else:
        voxel_pts = None

    # ------------------------------------------------------------------
    # Trace field lines & collect connection lengths
    # ------------------------------------------------------------------
    fieldlines = []
    Lpars = []

    for x0 in starts:
        pts, Lpar = trace_field_line_with_connection(
            x0, B_func, wall_mask_func, cfg, ds=ds, max_steps=fieldline_steps
        )
        fieldlines.append(pts)
        Lpars.append(Lpar)

    Lpars = np.array(Lpars)
    Lmin = float(Lpars.min())
    Lmax = float(Lpars.max() + 1e-12)
    cmap = plt.get_cmap("viridis")

    # ------------------------------------------------------------------
    # Plot: 3D geometry + L_parallel(theta)
    # ------------------------------------------------------------------
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(8.0, 4.8))
    gs = GridSpec(1, 2, width_ratios=[2.0, 1.0], figure=fig)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    axL = fig.add_subplot(gs[0, 1])

    # 1) Coils
    if coils is not None:
        coils.plot(
            ax=ax3d,
            show=False,
            color="saddlebrown",
            linewidth=3.0,
            label="Coils",
        )

    # 2) Toroidal SOL surface
    surf = ax3d.plot_surface(
        X_s,
        Y_s,
        Z_s,
        rstride=2,
        cstride=2,
        linewidth=0,
        antialiased=True,
        alpha=0.25,
        color="tab:blue",
        shade=True,
    )

    # 3) Optional voxel boundary points
    if voxel_pts is not None and voxel_pts.size > 0:
        ax3d.scatter(
            voxel_pts[:, 0],
            voxel_pts[:, 1],
            voxel_pts[:, 2],
            s=1.5,
            alpha=0.05,
            color="tab:blue",
        )

    # 4) Field lines colored by connection length
    for pts, Lpar in zip(fieldlines, Lpars):
        t = (Lpar - Lmin) / (Lmax - Lmin + 1e-12)
        color = cmap(t)
        ax3d.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            linewidth=1.6,
            alpha=0.95,
            color=color,
        )

    # Axes styling
    ax3d.set_xlabel(r"$x$")
    ax3d.set_ylabel(r"$y$")
    ax3d.set_zlabel(r"$z$")

    ax3d.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax3d.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax3d.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax3d.grid(False)

    max_range = max(
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min(),
    ) / 2.0
    xmid = 0.5 * (x.max() + x.min())
    ymid = 0.5 * (y.max() + y.min())
    zmid = 0.5 * (z.max() + z.min())
    ax3d.set_xlim(xmid - max_range, xmid + max_range)
    ax3d.set_ylim(ymid - max_range, ymid + max_range)
    ax3d.set_zlim(zmid - max_range, zmid + max_range)
    ax3d.view_init(elev=22, azim=-60)

    ax3d.set_title("ESSOS coils, field lines, and toroidal SOL surface")

    # Legend for coils + surface
    coil_handle = Line2D(
        [0], [0],
        color="saddlebrown",
        linewidth=3.0,
        label="Coils",
    )
    surf_handle = Patch(
        facecolor="tab:blue",
        edgecolor="none",
        alpha=0.25,
        label="SOL surface",
    )
    ax3d.legend(
        handles=[coil_handle, surf_handle],
        loc="upper left",
        frameon=False,
    )

    # Colorbar for connection length
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=Lmin, vmax=Lmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax3d,
        pad=0.02,
        shrink=0.8,
    )
    cbar.set_label(r"$L_\parallel$")

    # ------------------------------------------------------------------
    # Right panel: L_parallel vs poloidal angle theta
    # ------------------------------------------------------------------
    axL.plot(theta_pol, Lpars, "o-", ms=4)
    axL.set_xlabel(r"Poloidal angle $\theta$")
    axL.set_ylabel(r"$L_\parallel$")
    axL.set_xlim(0.0, 2.0 * np.pi)
    axL.set_xticks(
        [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    )
    axL.set_xticklabels(
        ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    )
    axL.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = os.path.join(outdir, "geometry_torus_Lpar_vs_theta.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)

    print(
        f"[fig] Saved torus geometry figure with L_parallel(theta) to '{fname}'. "
        f"(L_parallel in [{Lmin:.2f}, {Lmax:.2f}])"
    )


def toroidal_wall_mask(xyz: Array,
                       R0: float = 1.0,
                       a: float = 0.6) -> Array:
    """
    Toroidal plasma mask for a circular torus:

        (sqrt(x^2 + y^2) - R0)^2 + z^2 < a^2

    mask = 1 inside torus, 0 outside.
    R0 : major radius
    a  : minor radius (SOL thickness).
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    R = jnp.sqrt(x**2 + y**2)
    dist2 = (R - R0)**2 + z**2
    mask = (dist2 < a**2).astype(jnp.float64)
    return mask

def make_dynamics_figure(times: np.ndarray,
                         y_hist: np.ndarray,
                         cfg: SOLConfig,
                         wall_mask_func: Callable[[Array], Array],
                         outdir: str = "figures") -> None:
    """
    Summarize the time evolution of volume-averaged n, T_e and RMS u_parallel
    inside the plasma.

    Produces a 2-panel figure:
      top: <n>, <T_e> vs t
      bottom: <u_parallel^2>^{1/2} vs t
    """
    os.makedirs(outdir, exist_ok=True)

    # Rebuild mask on the same grid
    x, y, z, xyz = build_cartesian_grid(cfg)
    mask_plasma = np.array(wall_mask_func(xyz))
    mask = mask_plasma.astype(np.float64)

    Nt, _, Nx, Ny, Nz = y_hist.shape
    n_hist = y_hist[:, 0]
    u_hist = y_hist[:, 1]
    T_hist = y_hist[:, 2]

    # Broadcast mask to all times
    mask4d = mask[None, ...]  # (1, Nx,Ny,Nz)
    vol_norm = mask4d.sum()

    n_mean = (n_hist * mask4d).sum(axis=(1, 2, 3)) / vol_norm
    T_mean = (T_hist * mask4d).sum(axis=(1, 2, 3)) / vol_norm
    u_rms = np.sqrt(
        (u_hist**2 * mask4d).sum(axis=(1, 2, 3)) / vol_norm
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(5.0, 6.0)
    )

    # Top: density & temperature averages
    ax1.plot(times, n_mean, label=r"$\langle n \rangle$")
    ax1.plot(times, T_mean, label=r"$\langle T_e \rangle$")
    ax1.set_ylabel("Volume averages (arb. units)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Bottom: RMS parallel flow
    ax2.plot(times, u_rms, label=r"$\langle u_\parallel^2 \rangle^{1/2}$")
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel("RMS $u_\parallel$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    fname = os.path.join(outdir, "dynamics_volume_averages.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)

    print(f"[fig] Saved dynamics figure to '{fname}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Optional: limit XLA host devices
    number_of_processors_to_use = 1
    os.environ["XLA_FLAGS"] = (
        f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
    )

    cfg = SOLConfig(
        Nx=32,
        Ny=32,
        Nz=32,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        zlim=(-1.5, 1.5),
        D_perp=5e-3,
        chi_par=1.0,
        chi_perp=5e-3,
        mu_perp=5e-3,
        nu_par=0.5,
        gamma_e=5.0 / 3.0,
        n_wall=1e-4,
        T_wall=0.03,
        m_i=1.0,
        ds_par=0.05,
        t0=0.0,
        t1=2e-1,
        save_every=5e-2,
        rtol=1e-5,
        atol=1e-7,
    )

    # Torus geometry parameters (must fit inside xlim/ylim/zlim)
    R0 = 1.0   # major radius
    a  = 0.5   # minor radius (outer radius ~ R0 + a = 1.5)

    # Choose wall mask: circular torus
    wall_mask = partial(toroidal_wall_mask, R0=R0, a=a)

    # Try to use ESSOS; if unavailable, fall back to test B field
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        json_file = os.path.join(
            this_dir,
            "input_files",
            "ESSOS_biot_savart_LandremanPaulQA.json",
        )
        B_func, coils = make_essos_B_func(json_file)
        print(f"[main] Using ESSOS B field from '{json_file}'.")
    except Exception as e:
        print("[main] ESSOS not available or JSON missing. "
              "Falling back to simple test B field.")
        print(f"[main] Reason: {e}")

        @jit
        def B_func(xyz_flat: Array) -> Array:
            x = xyz_flat[:, 0]
            y = xyz_flat[:, 1]
            z = xyz_flat[:, 2]
            Bz0 = 1.0
            Bx = -y
            By = x
            Bz = jnp.ones_like(z) * Bz0
            return jnp.stack([Bx, By, Bz], axis=-1)

        coils = None

    # Build grid once for figure helper
    x, y, z, xyz = build_cartesian_grid(cfg)

    # Run simulation with toroidal mask
    times, y_hist = run_sol_fci_demo(
        cfg,
        B_func=B_func,
        wall_mask_func=wall_mask,
    )

    # Existing 2D slices / profiles
    make_publication_figures(times, y_hist, np.array(x), np.array(y), np.array(z))

    # 3D torus + L_parallel(theta) figure
    if coils is not None:
        make_3d_geometry_figure(
            cfg,
            coils,
            B_func,
            wall_mask_func=wall_mask,
            R0=R0,
            a=a,
            outdir="figures",
        )

    # New dynamics summary figure
    make_dynamics_figure(times, y_hist, cfg, wall_mask_func=wall_mask,
                         outdir="figures")

    print("[main] SOL FCI simulation and figure generation complete.")