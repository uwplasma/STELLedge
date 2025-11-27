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

from functools import partial

from plots import (make_3d_geometry_figure, make_dynamics_figure,
                   toroidal_wall_mask, make_essos_B_func, make_publication_figures)

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