#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sol_fci_essos_v3.py
===================

Flux-Coordinate Independent (FCI) SOL Transport Solver with ESSOS B-field
and 0D neutral model + 1D benchmark scalings
============================================

v3 adds to v2:
  • Circular-torus SOL mask (R0, a).
  • 3D connection-length field L_parallel(x).
  • Simple sheath-loss model:
        Γ_|| ~ n c_s / L_||
        q_|| ~ γ_s n T_e c_s
  • 0D neutral inventory:
        dn_n/dt = (f_rec Γ_loss - Ion_rate - n_n V/τ_pump)/V
    with ionization source S_ion = k_ion n_n n.
  • Ionization energy loss E_ion in T_e equation.
  • A D_perp scan and comparison to a 1D SOL benchmark:
        λ_n ~ sqrt(D_perp L_|| / c_s).
  • New figures: dynamics + neutrals, SOL-width scaling, etc.

This script depends on:
  • ESSOS (Biot–Savart for coils/B field)
  • Diffrax, JAX
  • plots.py in the same directory (geometry + diagnostics figures)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Tuple, List

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from jax import debug as jdbg
from jax import tree_util

import diffrax as dfx
import numpy as np

from plots import (
    make_3d_geometry_figure,
    make_dynamics_figure,
    make_publication_figures,
    make_sol_width_scaling_figure,
    measure_sol_width_and_theory,
    toroidal_wall_mask,
    make_essos_B_func,
)

Array = jnp.ndarray


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SOLConfig:
    """
    Configuration for the SOL FCI solver (dimensionless units).

    Geometry
    --------
    R0 : major radius, a : minor radius of circular torus.

    Transport
    ---------
    D_perp : cross-field particle diffusion.
    chi_par : parallel thermal diffusivity (Spitzer–Härm-like).
    chi_perp : cross-field thermal diffusivity.
    mu_perp : cross-field viscosity for u_parallel.
    nu_par : parallel friction/damping for u_parallel.
    gamma_e : parallel "adiabatic index" for electrons.

    Sheath & neutrals
    -----------------
    gamma_sheath : sheath heat-transmission factor.
    f_recycle : fraction of ion losses turned into neutrals.
    L_min : floor for L_parallel.
    tau_pump : neutral pumping time.
    k_ion : ionization rate coefficient (dimensionless).
    E_ion : effective ionization energy (in T_e units).

    Grid & FCI
    ----------
    Nx,Ny,Nz and xlim,ylim,zlim define a Cartesian box around the torus.
    ds_par : parallel step for FCI derivative stencil.

    Time integration
    ----------------
    t0, t1, save_every, rtol, atol.
    """
    Nx: int = 32
    Ny: int = 32
    Nz: int = 32
    xlim: Tuple[float, float] = (-1.5, 1.5)
    ylim: Tuple[float, float] = (-1.5, 1.5)
    zlim: Tuple[float, float] = (-1.5, 1.5)

    # Torus geometry
    R0: float = 1.0
    a: float = 0.5

    # Transport
    D_perp: float = 5e-3
    chi_par: float = 1.0
    chi_perp: float = 5e-3
    mu_perp: float = 5e-3
    nu_par: float = 0.5
    gamma_e: float = 5.0 / 3.0

    # Sheath + neutrals
    gamma_sheath: float = 7.0
    f_recycle: float = 0.9
    L_min: float = 0.1
    tau_pump: float = 1.0       # neutral pumping time
    k_ion: float = 1.0          # dimensionless ionization rate coefficient
    E_ion: float = 0.02         # ionization energy in "T units"

    # Wall / plasma
    n_wall: float = 1e-4
    T_wall: float = 0.03
    m_i: float = 1.0

    # FCI
    ds_par: float = 0.05

    # Time integration
    t0: float = 0.0
    t1: float = 2e-1
    save_every: float = 5e-2
    rtol: float = 1e-5
    atol: float = 1e-7


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def build_cartesian_grid(cfg: SOLConfig) -> Tuple[Array, Array, Array, Array]:
    """Build 3D Cartesian grid and coordinate array."""
    x = jnp.linspace(cfg.xlim[0], cfg.xlim[1], cfg.Nx)
    y = jnp.linspace(cfg.ylim[0], cfg.ylim[1], cfg.Ny)
    z = jnp.linspace(cfg.zlim[0], cfg.zlim[1], cfg.Nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    xyz = jnp.stack([X, Y, Z], axis=-1)
    return x, y, z, xyz


def example_wall_mask(xyz: Array, R_wall: float = 1.2) -> Array:
    """Fallback cylindrical mask."""
    x = xyz[..., 0]
    y = xyz[..., 1]
    R = jnp.sqrt(x**2 + y**2)
    return (R < R_wall).astype(jnp.float64)


# ---------------------------------------------------------------------------
# Trilinear interpolation & FCI derivatives
# ---------------------------------------------------------------------------

def _trilinear_interp(F: Array,
                      x_grid: Array,
                      y_grid: Array,
                      z_grid: Array,
                      pts: Array) -> Array:
    """Trilinear interpolation on a 3D Cartesian grid."""
    Nx, Ny, Nz = F.shape
    x0, y0, z0 = x_grid[0], y_grid[0], z_grid[0]
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]

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

    f000 = F[ix0, iy0, iz0]
    f001 = F[ix0, iy0, iz1]
    f010 = F[ix0, iy1, iz0]
    f011 = F[ix0, iy1, iz1]
    f100 = F[ix1, iy0, iz0]
    f101 = F[ix1, iy0, iz1]
    f110 = F[ix1, iy1, iz0]
    f111 = F[ix1, iy1, iz1]

    c00 = f000 * (1 - tx) + f100 * tx
    c01 = f001 * (1 - tx) + f101 * tx
    c10 = f010 * (1 - tx) + f110 * tx
    c11 = f011 * (1 - tx) + f111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    vals = c0 * (1 - tz) + c1 * tz
    return vals


def trilinear_interp(F, x_grid, y_grid, z_grid, pts):
    return _trilinear_interp(F, x_grid, y_grid, z_grid, pts)


def fci_parallel_derivs(field: Array,
                        xyz: Array,
                        b_grid: Array,
                        x_grid: Array,
                        y_grid: Array,
                        z_grid: Array,
                        ds: float) -> Tuple[Array, Array]:
    """
    Hariri-style FCI derivatives:
        f±(x) = f(x ± ds b_hat)
        ∂∥ f ≈ (f+ - f-) / (2 ds)
        ∂∥² f ≈ (f+ - 2f0 + f-) / ds²
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
# Perpendicular Laplacian
# ---------------------------------------------------------------------------

def second_derivatives_cartesian(F: Array,
                                 dx: float,
                                 dy: float,
                                 dz: float) -> Tuple[Array, Array, Array, Array]:
    """Second derivatives + Laplacian with periodic BCs."""
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
    """Compute ∇⊥² F = ∇²F - ∂∥²F."""
    _, _, _, lap = second_derivatives_cartesian(F, dx, dy, dz)
    dpar_f, d2par_f = fci_parallel_derivs(F, xyz, b_grid, x_grid, y_grid, z_grid, ds)
    lap_perp = lap - d2par_f
    return lap_perp, dpar_f, d2par_f


# ---------------------------------------------------------------------------
# Connection length + recycling weight (pre-processing, numpy side)
# ---------------------------------------------------------------------------

def compute_connection_length_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xyz: np.ndarray,
    mask_plasma: np.ndarray,
    b_grid: np.ndarray,
    ds_geom: float,
    max_steps: int = 512,
) -> np.ndarray:
    """
    Approximate connection length L_parallel(x) on the grid.

    For each plasma cell, march along local b_hat in ± directions with fixed
    step ds_geom until we exit the mask; total arclength is L_parallel.
    """
    Nx, Ny, Nz, _ = xyz.shape
    L_conn = np.zeros((Nx, Ny, Nz), dtype=np.float64)

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])

    def idx_from_pos(pos):
        ix = int(np.round((pos[0] - x[0]) / dx))
        iy = int(np.round((pos[1] - y[0]) / dy))
        iz = int(np.round((pos[2] - z[0]) / dz))
        if ix < 0 or ix >= Nx or iy < 0 or iy >= Ny or iz < 0 or iz >= Nz:
            return None
        return ix, iy, iz

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if mask_plasma[i, j, k] < 0.5:
                    continue
                b_local = b_grid[i, j, k, :]
                b_norm = np.linalg.norm(b_local)
                if b_norm < 1e-8:
                    continue
                b_hat = b_local / b_norm
                L_tot = 0.0
                x0 = xyz[i, j, k, :]

                for sgn in (+1.0, -1.0):
                    pos = x0.copy()
                    for _ in range(max_steps):
                        pos = pos + sgn * ds_geom * b_hat
                        idx = idx_from_pos(pos)
                        if idx is None:
                            break
                        ii, jj, kk = idx
                        if mask_plasma[ii, jj, kk] < 0.5:
                            break
                        L_tot += ds_geom

                L_conn[i, j, k] = L_tot

    return L_conn


def make_recycling_weight(
    xyz: Array,
    mask_plasma: Array,
    cfg: SOLConfig,
    width_r: float | None = None,
    width_z: float | None = None,
) -> Array:
    """
    Weight function w_recycle(x) localized near the outer midplane, used
    to visualize where ionization of neutrals preferentially occurs.
    """
    if width_r is None:
        width_r = 0.15 * cfg.a
    if width_z is None:
        width_z = 0.3 * cfg.a

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    R = jnp.sqrt(x**2 + y**2)
    r_minor = jnp.sqrt((R - cfg.R0) ** 2 + z**2)

    w = jnp.exp(
        -((r_minor - cfg.a) ** 2 / (2 * width_r**2) + z**2 / (2 * width_z**2))
    )
    w = w * mask_plasma
    w_sum = jnp.sum(w)
    w_norm = jnp.where(w_sum > 0.0, w / w_sum, jnp.zeros_like(w))
    return w_norm


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
    vol_cell: float
    V_plasma: float
    L_conn: Array           # (Nx, Ny, Nz)
    w_recycle: Array        # (Nx, Ny, Nz)
    S_n: Array              # external density source
    S_T: Array              # external heat source

    def tree_flatten(self):
        children = (
            self.x_grid,
            self.y_grid,
            self.z_grid,
            self.xyz,
            self.b_grid,
            self.mask_plasma,
            self.L_conn,
            self.w_recycle,
            self.S_n,
            self.S_T,
        )
        aux_data = (
            self.cfg,
            self.dx,
            self.dy,
            self.dz,
            self.vol_cell,
            self.V_plasma,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            cfg,
            dx,
            dy,
            dz,
            vol_cell,
            V_plasma,
        ) = aux_data
        (
            x_grid,
            y_grid,
            z_grid,
            xyz,
            b_grid,
            mask_plasma,
            L_conn,
            w_recycle,
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
            vol_cell=vol_cell,
            V_plasma=V_plasma,
            L_conn=L_conn,
            w_recycle=w_recycle,
            S_n=S_n,
            S_T=S_T,
        )


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def make_sources(xyz: Array, cfg: SOLConfig) -> Tuple[Array, Array]:
    """
    External sources S_n and S_T.

    Here: a Gaussian near the outer midplane (e.g. RF / neutral heating).
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    R = jnp.sqrt(x**2 + y**2)
    R0_src = cfg.R0 + 0.3 * cfg.a
    z0 = 0.0
    width_R = 0.3 * cfg.a
    width_z = 0.5 * cfg.a

    S_n0 = 1.0
    S_T0 = 10.0

    gauss = jnp.exp(
        -((R - R0_src) ** 2 / (2 * width_R**2) + (z - z0) ** 2 / (2 * width_z**2))
    )
    S_n = S_n0 * gauss
    S_T = S_T0 * gauss
    return S_n, S_T


# ---------------------------------------------------------------------------
# RHS with 0D neutrals + sheath
# ---------------------------------------------------------------------------

def rhs_sol(t: float, y_flat: Array, params: SOLParams) -> Array:
    """
    RHS of the coupled SOL + 0D neutrals system.

    State vector:
        y_flat = [n_flat, u_flat, T_flat, n_n]

    where n_n is a volume-averaged neutral density.
    """
    cfg = params.cfg
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    N = Nx * Ny * Nz

    n_flat = y_flat[0:N]
    u_flat = y_flat[N : 2 * N]
    T_flat = y_flat[2 * N : 3 * N]
    n_n = y_flat[3 * N]  # scalar neutral density

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
    vol_cell = params.vol_cell
    V_plasma = params.V_plasma

    # Apply wall Dirichlet conditions
    n = mask * n + (1.0 - mask) * cfg.n_wall
    T = mask * T + (1.0 - mask) * cfg.T_wall
    u = mask * u  # u_wall = 0

    # Basic ranges
    jdbg.print(
        "[rhs] t={t:.3e}  n:[{nmin:.3e},{nmax:.3e}]  "
        "T:[{Tmin:.3e},{Tmax:.3e}]  u:[{umin:.3e},{umax:.3e}]  n_n={nn:.3e}",
        t=t,
        nmin=jnp.min(n),
        nmax=jnp.max(n),
        Tmin=jnp.min(T),
        Tmax=jnp.max(T),
        umin=jnp.min(u),
        umax=jnp.max(u),
        nn=n_n,
    )

    # Geometry / sheath
    L_safe = jnp.maximum(params.L_conn, cfg.L_min)
    c_s = jnp.sqrt(T / cfg.m_i) * mask

    # ------------------------------------------------------------------
    # Continuity: dn/dt
    # ------------------------------------------------------------------
    n_u = n * u
    lap_perp_n, _, _ = perp_laplacian(
        n, xyz, b_grid, x_grid, y_grid, z_grid, cfg.ds_par, dx, dy, dz
    )
    dpar_nu, _ = fci_parallel_derivs(
        n_u, xyz, b_grid, x_grid, y_grid, z_grid, cfg.ds_par
    )

    # Baseline
    dn_dt = -dpar_nu + cfg.D_perp * lap_perp_n + params.S_n

    # Sheath particle loss
    S_loss_n = -n * c_s / L_safe

    # Ionization source from neutrals (0D n_n)
    S_ion = cfg.k_ion * n_n * n * mask

    dn_dt = dn_dt + S_loss_n + S_ion
    dn_dt = mask * dn_dt

    # Integrated rates for neutral ODE
    Gamma_loss = -jnp.sum(S_loss_n * mask) * vol_cell     # total ion loss
    Ion_rate_total = jnp.sum(S_ion * mask) * vol_cell      # total ionization

    # ------------------------------------------------------------------
    # Parallel momentum: du/dt
    # ------------------------------------------------------------------
    lap_perp_u, dpar_u, _ = perp_laplacian(
        u, xyz, b_grid, x_grid, y_grid, z_grid, cfg.ds_par, dx, dy, dz
    )
    p = n * T
    dpar_p, _ = fci_parallel_derivs(
        p, xyz, b_grid, x_grid, y_grid, z_grid, cfg.ds_par
    )

    n_safe = jnp.maximum(n, 1e-8)
    du_dt = (
        -u * dpar_u
        - (1.0 / cfg.m_i) * (1.0 / n_safe) * dpar_p
        - cfg.nu_par * u
        + cfg.mu_perp * lap_perp_u
        - c_s * u / L_safe
    )
    du_dt = mask * du_dt

    # ------------------------------------------------------------------
    # Electron temperature: dT/dt
    # ------------------------------------------------------------------
    lap_perp_T, dpar_T, d2par_T = perp_laplacian(
        T, xyz, b_grid, x_grid, y_grid, z_grid, cfg.ds_par, dx, dy, dz
    )

    dT_dt = (
        -u * dpar_T
        - (cfg.gamma_e - 1.0) * T * dpar_u
        + cfg.chi_par * d2par_T
        + cfg.chi_perp * lap_perp_T
        + params.S_T / (1.5 * n_safe)
        - (2.0 / 3.0) * cfg.gamma_sheath * T * c_s / L_safe
        - cfg.E_ion * S_ion / (1.5 * n_safe)  # ionization energy sink
    )
    dT_dt = mask * dT_dt

    # ------------------------------------------------------------------
    # Neutral ODE (0D)
    # ------------------------------------------------------------------
    dn_n_dt = (
        cfg.f_recycle * Gamma_loss
        - Ion_rate_total
        - n_n * V_plasma / cfg.tau_pump
    ) / (V_plasma + 1e-16)

    # Assemble flattened derivative
    dy_dt = jnp.concatenate(
        [
            dn_dt.reshape(-1),
            du_dt.reshape(-1),
            dT_dt.reshape(-1),
            jnp.array([dn_n_dt]),
        ],
        axis=0,
    )

    jdbg.print(
        "[rhs] Γ_loss={g_loss:.3e}, Ion_rate={ion:.3e}, dn_n_dt={dnn:.3e}",
        g_loss=Gamma_loss,
        ion=Ion_rate_total,
        dnn=dn_n_dt,
    )

    return dy_dt


# ---------------------------------------------------------------------------
# Main driver: build params, run Diffrax, and make figures
# ---------------------------------------------------------------------------

def run_sol_fci_demo(
    cfg: SOLConfig,
    B_func: Callable[[Array], Array],
    wall_mask_func: Callable[[Array], Array],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Run a single 3D SOL FCI simulation.

    Returns
    -------
    times       : (Nt,)
    y_hist      : (Nt, 3, Nx, Ny, Nz)  [n,u,T]
    n_neut_hist : (Nt,)
    x,y,z       : 1D grids
    mask_plasma : (Nx,Ny,Nz)
    L_conn      : (Nx,Ny,Nz)
    """
    print("=== SOL FCI ESSOS-based Solver Demo (v3) ===")
    print(cfg)

    # Build grid
    x, y, z, xyz = build_cartesian_grid(cfg)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    vol_cell = dx * dy * dz

    print(f"[setup] Grid: Nx={cfg.Nx}, Ny={cfg.Ny}, Nz={cfg.Nz}")
    print(f"[setup] dx={dx:.3e}, dy={dy:.3e}, dz={dz:.3e}")

    # Magnetic field on grid
    xyz_flat = xyz.reshape((-1, 3))
    B_flat = B_func(xyz_flat)
    B_grid = B_flat.reshape((cfg.Nx, cfg.Ny, cfg.Nz, 3))

    B_mag = jnp.sqrt(jnp.sum(B_grid**2, axis=-1) + 1e-16)
    b_grid = B_grid / B_mag[..., None]

    print(
        "[setup] B field computed on grid. "
        f"<|B|>={float(jnp.mean(B_mag)):.3e}, "
        f"min={float(jnp.min(B_mag)):.3e}, max={float(jnp.max(B_mag)):.3e}"
    )

    # Wall / plasma mask
    mask_plasma = wall_mask_func(xyz)
    V_plasma = float(jnp.sum(mask_plasma) * vol_cell)
    print("[setup] Plasma mask constructed from toroidal wall.")
    print(f"[setup] Plasma volume fraction: {float(jnp.mean(mask_plasma)):.3f}")

    # Connection length
    print("[setup] Computing connection-length field L_parallel(x)...")
    L_conn_np = compute_connection_length_field(
        np.array(x),
        np.array(y),
        np.array(z),
        np.array(xyz),
        np.array(mask_plasma),
        np.array(b_grid),
        ds_geom=float(cfg.ds_par),
        max_steps=512,
    )
    L_conn = jnp.asarray(L_conn_np)

    # Recycling localization weight
    w_recycle = make_recycling_weight(xyz, mask_plasma, cfg)

    # Sources
    S_n, S_T = make_sources(xyz, cfg)
    print("[setup] Sources S_n and S_T initialized.")

    # Initial conditions
    n0 = cfg.n_wall + mask_plasma * 0.9
    u0 = jnp.zeros_like(n0)
    R = jnp.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
    T0 = cfg.T_wall + mask_plasma * 0.5 * jnp.exp(
        -((R - (cfg.R0 + 0.5 * cfg.a)) ** 2 / (2 * (0.2 * cfg.a) ** 2))
    )
    n_n0 = 1e-3  # initial neutral density (dimensionless)

    print(
        f"[IC] n0:[{float(jnp.min(n0)):.3e},{float(jnp.max(n0)):.3e}] "
        f"T0:[{float(jnp.min(T0)):.3e},{float(jnp.max(T0)):.3e}] "
        f"n_n0={float(n_n0):.3e}"
    )

    N = cfg.Nx * cfg.Ny * cfg.Nz
    y0 = jnp.concatenate(
        [n0.reshape(-1), u0.reshape(-1), T0.reshape(-1), jnp.array([n_n0])],
        axis=0,
    )

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
        vol_cell=vol_cell,
        V_plasma=V_plasma,
        L_conn=L_conn,
        w_recycle=w_recycle,
        S_n=S_n,
        S_T=S_T,
    )

    term = dfx.ODETerm(rhs_sol)
    save_ts = jnp.arange(cfg.t0, cfg.t1 + 1e-12, cfg.save_every)
    saveat = dfx.SaveAt(ts=save_ts, t0=True, t1=True)

    solver = dfx.Tsit5()
    stepsize_controller = dfx.PIDController(rtol=cfg.rtol, atol=cfg.atol)

    print("[time] Starting Diffrax solve (Tsit5 explicit)...")
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=cfg.t0,
        t1=cfg.t1,
        dt0=1e-5,
        y0=y0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=1_000_000,
    )
    print("[time] Diffrax solve complete.")

    times = np.array(sol.ts)
    ys = np.array(sol.ys)  # (Nt, 3N+1)

    Nt = ys.shape[0]
    plasma_flat = ys[:, : 3 * N]
    y_hist = plasma_flat.reshape((Nt, 3, cfg.Nx, cfg.Ny, cfg.Nz))
    n_neut_hist = ys[:, 3 * N]

    # Final diagnostics
    n_final = y_hist[-1, 0]
    u_final = y_hist[-1, 1]
    T_final = y_hist[-1, 2]
    print(
        f"[final] n:[{float(n_final.min()):.3e},{float(n_final.max()):.3e}], "
        f"T:[{float(T_final.min()):.3e},{float(T_final.max()):.3e}], "
        f"u:[{float(u_final.min()):.3e},{float(u_final.max()):.3e}], "
        f"n_n_final={float(n_neut_hist[-1]):.3e}"
    )

    return (
        times,
        y_hist,
        n_neut_hist,
        np.array(x),
        np.array(y),
        np.array(z),
        np.array(mask_plasma),
        np.array(L_conn),
    )


# ---------------------------------------------------------------------------
# D_perp scan + 1D benchmark
# ---------------------------------------------------------------------------

def run_D_perp_scan(
    cfg_base: SOLConfig,
    B_func: Callable[[Array], Array],
    wall_mask_func: Callable[[Array], Array],
    D_values: List[float],
    outdir: str = "figures",
):
    """
    Run a short SOL evolution for several D_perp values and compare the
    measured scrape-off width λ_n to the 1D analytic prediction.
    """
    lambda_num = []
    lambda_th = []

    for D in D_values:
        print("\n=== D_perp scan entry: D_perp = ", D, " ===")
        cfg = replace(cfg_base, D_perp=D, t1=0.1, save_every=0.05)
        (
            times,
            y_hist,
            n_neut_hist,
            x,
            y,
            z,
            mask_plasma,
            L_conn,
        ) = run_sol_fci_demo(cfg, B_func, wall_mask_func)

        n_final = y_hist[-1, 0]
        T_final = y_hist[-1, 2]
        lam_n, lam_th = measure_sol_width_and_theory(
            x, y, z, n_final, T_final, mask_plasma, L_conn, cfg
        )
        lambda_num.append(lam_n)
        lambda_th.append(lam_th)

    make_sol_width_scaling_figure(
        np.array(D_values),
        np.array(lambda_num),
        np.array(lambda_th),
        outdir=outdir,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Optional: limit XLA host devices
    number_of_processors_to_use = 1
    os.environ["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={number_of_processors_to_use}"
    )

    cfg = SOLConfig()

    # Toroidal wall mask based on (R0, a)
    wall_mask = partial(toroidal_wall_mask, R0=cfg.R0, a=cfg.a)

    # Try to use ESSOS; if unavailable, fall back to simple test B field
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
        print(
            "[main] ESSOS not available or JSON missing. "
            "Falling back to simple test B field."
        )
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

    # ------------------------------------------------------------------
    # Single "flagship" run with full diagnostics
    # ------------------------------------------------------------------
    (
        times,
        y_hist,
        n_neut_hist,
        x,
        y,
        z,
        mask_plasma,
        L_conn,
    ) = run_sol_fci_demo(cfg, B_func, wall_mask)

    # 2D slices + radial profiles
    make_publication_figures(times, y_hist, x, y, z, outdir="figures")

    # Dynamics (volume-averaged) + neutrals
    make_dynamics_figure(
        times,
        y_hist,
        n_neut_hist,
        cfg,
        mask_plasma,
        outdir="figures",
    )

    # 3D geometry + L_parallel panel
    if coils is not None:
        make_3d_geometry_figure(
            cfg,
            coils,
            B_func,
            wall_mask_func=wall_mask,
            outdir="figures",
        )

    # ------------------------------------------------------------------
    # D_perp scan vs 1D SOL scaling
    # ------------------------------------------------------------------
    D_values = [2e-3, 5e-3, 1e-2, 2e-2]
    run_D_perp_scan(cfg, B_func, wall_mask, D_values, outdir="figures")

    print("[main] SOL FCI v3 simulation and figure generation complete.")
