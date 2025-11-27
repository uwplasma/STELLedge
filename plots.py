#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots.py
========

Plotting utilities for sol_fci_essos_v3.py:

  • toroidal_wall_mask
  • ESSOS wrapper
  • 3D geometry figure: coils + torus + field lines + L_parallel
  • 2D slices and radial profiles
  • Dynamics figure (volume-averaged n, T, u, neutrals)
  • SOL-width scaling vs 1D benchmark
"""

from __future__ import annotations

import os
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

Array = jnp.ndarray


# ---------------------------------------------------------------------------
# Basic geometry helpers
# ---------------------------------------------------------------------------

def toroidal_wall_mask(xyz: Array, R0: float, a: float) -> Array:
    """
    Circular-torus mask:

        r_minor = sqrt[(R - R0)^2 + z^2]
        mask = 1 for r_minor < a, else 0
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    R = jnp.sqrt(x**2 + y**2)
    r_minor = jnp.sqrt((R - R0) ** 2 + z**2)
    return (r_minor < a).astype(jnp.float64)


def make_essos_B_func(json_file: str):
    """
    Build a JAX-compatible B_func(xyz_flat) using ESSOS BiotSavart.

    Returns
    -------
    B_func : callable  (xyz_flat: (N,3) -> (N,3))
    coils  : ESSOS coils object (for plotting)
    """
    from essos.fields import BiotSavart
    from essos.coils import Coils_from_json

    coils = Coils_from_json(json_file)
    field = BiotSavart(coils)

    @jit
    def B_func(xyz_flat: Array) -> Array:
        return vmap(field.B_contravariant)(xyz_flat)

    return B_func, coils


def build_cartesian_grid_from_cfg(cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Local copy of the grid builder to avoid import cycles."""
    x = np.linspace(cfg.xlim[0], cfg.xlim[1], cfg.Nx)
    y = np.linspace(cfg.ylim[0], cfg.ylim[1], cfg.Ny)
    z = np.linspace(cfg.zlim[0], cfg.zlim[1], cfg.Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    xyz = np.stack([X, Y, Z], axis=-1)
    return x, y, z, xyz


# ---------------------------------------------------------------------------
# Field-line tracing (for geometry figure)
# ---------------------------------------------------------------------------

def trace_field_line(
    x0: np.ndarray,
    B_func: Callable[[Array], Array],
    ds: float,
    n_steps: int,
    mask_func: Callable[[Array], Array] | None = None,
) -> Tuple[np.ndarray, float]:
    """
    Trace a single field line starting at x0, stepping along +b with step ds.

    If mask_func is provided, stop when we leave the plasma (mask < 0.5).

    Returns
    -------
    pts : (n_pts,3)  coordinates along the line
    L   : total arclength inside mask (connection length along this line)
    """
    pts = np.zeros((n_steps, 3))
    pts[0] = x0
    L = 0.0

    for k in range(n_steps - 1):
        xk = jnp.asarray(pts[k])[None, :]
        Bk = B_func(xk)
        Bk_np = np.array(Bk)[0]
        Bmag = np.linalg.norm(Bk_np)
        if Bmag < 1e-8:
            pts = pts[: k + 1]
            break
        b = Bk_np / Bmag
        x_next = pts[k] + ds * b
        if mask_func is not None:
            val = np.array(mask_func(jnp.asarray(x_next)[None, :]))[0]
            if val < 0.5:
                pts = pts[: k + 1]
                break
        pts[k + 1] = x_next
        L += ds

    return pts, L


# ---------------------------------------------------------------------------
# 3D geometry figure
# ---------------------------------------------------------------------------

def make_3d_geometry_figure(
    cfg,
    coils,
    B_func: Callable[[Array], Array],
    wall_mask_func: Callable[[Array], Array],
    n_fieldlines: int = 24,
    fieldline_length: float = 4.0,
    fieldline_steps: int = 400,
    outdir: str = "figures",
) -> None:
    """
    3D torus + coils + field lines, colored by L_parallel along each line,
    plus a side panel with L_parallel vs poloidal angle.
    """
    os.makedirs(outdir, exist_ok=True)

    x, y, z, xyz = build_cartesian_grid_from_cfg(cfg)
    xyz_j = jnp.asarray(xyz)
    mask_plasma = np.array(wall_mask_func(xyz_j))

    # ring of start points near outer midplane
    theta = np.linspace(0.0, 2.0 * np.pi, n_fieldlines, endpoint=False)
    r0 = cfg.a * 0.8
    sgn = 1.0  # outboard
    R_start = cfg.R0 + r0 * np.cos(theta)
    Z_start = r0 * np.sin(theta)
    starts = np.stack(
        [R_start * np.cos(0.0), R_start * np.sin(0.0), Z_start],
        axis=1,
    )

    def mask_point(xp: Array) -> Array:
        # xp: (1,3)
        return wall_mask_func(xp)

    ds = fieldline_length / fieldline_steps

    fieldlines = []
    L_lines = []
    theta_lines = []

    for t0, x0 in zip(theta, starts):
        pts, L = trace_field_line(x0, B_func, ds=ds, n_steps=fieldline_steps, mask_func=mask_point)
        fieldlines.append(pts)
        L_lines.append(L)
        theta_lines.append(t0)

    L_lines = np.array(L_lines)
    theta_lines = np.array(theta_lines)

    # Torus surface for SOL
    n_theta_surf = 128
    n_phi = 128
    th = np.linspace(0, 2 * np.pi, n_theta_surf, endpoint=False)
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    Xs = (cfg.R0 + cfg.a * np.cos(TH)) * np.cos(PH)
    Ys = (cfg.R0 + cfg.a * np.cos(TH)) * np.sin(PH)
    Zs = cfg.a * np.sin(TH)

    # --- Plot ---
    fig = plt.figure(figsize=(8.0, 4.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    # coils
    if coils is not None:
        coils.plot(ax=ax3d, show=False, color="saddlebrown", linewidth=2.5)

    # torus surface
    ax3d.plot_surface(
        Xs,
        Ys,
        Zs,
        rstride=3,
        cstride=3,
        linewidth=0,
        antialiased=True,
        alpha=0.25,
        color="tab:blue",
        shade=True,
    )

    # field lines colored by L_parallel
    L_min = L_lines.min()
    L_max = L_lines.max()
    norm = plt.Normalize(L_min, L_max)
    cmap = plt.cm.viridis

    for pts, L in zip(fieldlines, L_lines):
        ax3d.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            linewidth=1.8,
            color=cmap(norm(L)),
        )

    ax3d.set_xlabel(r"$x$")
    ax3d.set_ylabel(r"$y$")
    ax3d.set_zlabel(r"$z$")
    ax3d.set_title("ESSOS coils, field lines,\n and toroidal SOL surface")

    # equal aspect
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

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3d, pad=0.05)
    cbar.set_label(r"$L_\parallel$")

    # L_parallel vs poloidal angle
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(theta_lines, L_lines, "o-")
    ax2.set_xlabel(r"Poloidal angle $\theta$")
    ax2.set_ylabel(r"$L_\parallel(\theta)$")
    ax2.set_title(r"Connection length along SOL ring")
    ax2.grid(True)

    fig.tight_layout()
    fname = os.path.join(outdir, "geometry_coils_fieldlines_torus.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"[fig] Saved 3D geometry figure to '{fname}'.")


# ---------------------------------------------------------------------------
# 2D slices & radial profiles (existing figure)
# ---------------------------------------------------------------------------

def make_publication_figures(
    times: np.ndarray,
    y_hist: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    outdir: str = "figures",
) -> None:
    """
    1) T_e(x,y) slice at midplane.
    2) n(x,y) slice at midplane.
    3) Radial profiles at outer midplane.
    """
    os.makedirs(outdir, exist_ok=True)

    Nt, _, Nx, Ny, Nz = y_hist.shape
    n = y_hist[-1, 0]
    T = y_hist[-1, 2]

    iz0 = int(np.argmin(np.abs(z)))
    z0 = float(z[iz0])

    n_slice = n[:, :, iz0]
    T_slice = T[:, :, iz0]

    X, Y = np.meshgrid(x, y, indexing="ij")

    # Te slice
    fig1, ax1 = plt.subplots(figsize=(5.0, 4.0))
    c1 = ax1.pcolormesh(X, Y, T_slice.T, shading="auto")
    fig1.colorbar(c1, ax=ax1, label=r"$T_e$")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_title(rf"$T_e(x,y,z\approx {z0:.2f})$ at $t={times[-1]:.3e}$")
    fig1.tight_layout()
    fig1.savefig(os.path.join(outdir, "Te_slice_midplane.png"), dpi=300)
    plt.close(fig1)

    # n slice
    fig2, ax2 = plt.subplots(figsize=(5.0, 4.0))
    c2 = ax2.pcolormesh(X, Y, n_slice.T, shading="auto")
    fig2.colorbar(c2, ax=ax2, label=r"$n$")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")
    ax2.set_title(rf"$n(x,y,z\approx {z0:.2f})$ at $t={times[-1]:.3e}$")
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "n_slice_midplane.png"), dpi=300)
    plt.close(fig2)

    # radial profiles at outer midplane y>0
    iy_pos = np.argmax(y > 0.0) if np.any(y > 0.0) else Ny // 2
    n_prof = n[:, iy_pos, iz0]
    T_prof = T[:, iy_pos, iz0]
    R_prof = np.sqrt(x**2 + y[iy_pos] ** 2)

    fig3, ax3 = plt.subplots(figsize=(5.0, 4.0))
    ax3.plot(R_prof, T_prof, label=r"$T_e$")
    ax3.plot(R_prof, n_prof, label=r"$n$")
    ax3.set_xlabel(r"$R$")
    ax3.set_ylabel("Profiles (arb.)")
    ax3.set_title(rf"Radial profiles at $(y={y[iy_pos]:.2f}, z={z0:.2f})$")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(outdir, "radial_profiles.png"), dpi=300)
    plt.close(fig3)

    print(f"[fig] Saved midplane & profile figures to '{outdir}'.")


# ---------------------------------------------------------------------------
# Dynamics + neutrals
# ---------------------------------------------------------------------------

def make_dynamics_figure(
    times: np.ndarray,
    y_hist: np.ndarray,
    n_neut_hist: np.ndarray,
    cfg,
    mask_plasma_np: np.ndarray,
    outdir: str = "figures",
) -> None:
    """
    Volume-averaged evolution of n, T, u and neutrals.
    """
    os.makedirs(outdir, exist_ok=True)

    x, y, z, xyz = build_cartesian_grid_from_cfg(cfg)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    vol_cell = dx * dy * dz
    mask = mask_plasma_np

    Nt = times.size
    n_avg = np.zeros(Nt)
    T_avg = np.zeros(Nt)
    u_rms = np.zeros(Nt)

    for it in range(Nt):
        n = y_hist[it, 0]
        u = y_hist[it, 1]
        T = y_hist[it, 2]

        w = mask * vol_cell
        V = w.sum()

        n_avg[it] = (n * w).sum() / V
        T_avg[it] = (T * w).sum() / V
        u_rms[it] = np.sqrt(((u**2) * w).sum() / V)

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharex=True)

    ax = axes[0, 0]
    ax.plot(times, n_avg)
    ax.set_ylabel(r"$\langle n \rangle$")
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(times, T_avg)
    ax.set_ylabel(r"$\langle T_e \rangle$")
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(times, u_rms)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$u_\parallel^{\mathrm{rms}}$")
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(times, n_neut_hist)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$n_n$ (0D)")
    ax.grid(True)

    fig.suptitle("SOL volume-averaged dynamics and neutrals")
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fname = os.path.join(outdir, "dynamics_neutrals.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"[fig] Saved dynamics figure to '{fname}'.")


# ---------------------------------------------------------------------------
# 1D SOL benchmark + scaling
# ---------------------------------------------------------------------------

def measure_sol_width_and_theory(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n3d: np.ndarray,
    T3d: np.ndarray,
    mask_plasma: np.ndarray,
    L_conn: np.ndarray,
    cfg,
) -> Tuple[float, float]:
    """
    Measure SOL scrape-off width λ_n from the 3D simulation at midplane and
    compare with a 1D analytic λ ~ sqrt(D_perp L_parallel / c_s).
    """
    Nz = z.size
    iz0 = int(np.argmin(np.abs(z)))
    iy_pos = np.argmax(y > 0.0) if np.any(y > 0.0) else len(y) // 2

    n_line = n3d[:, iy_pos, iz0]
    T_line = T3d[:, iy_pos, iz0]
    mask_line = mask_plasma[:, iy_pos, iz0]
    L_line = L_conn[:, iy_pos, iz0]

    R_line = np.sqrt(x**2 + y[iy_pos] ** 2)
    r_minor = R_line - cfg.R0  # approx radial coord at midplane

    # use only SOL region inside plasma and r>0
    sel = (mask_line > 0.5) & (r_minor > 0.0)
    r = r_minor[sel]
    n_sol = n_line[sel]
    T_sol = T_line[sel]
    L_sol = L_line[sel]

    # avoid zeros
    n_sol = np.maximum(n_sol, 1e-8)

    # exponential fit: n ~ exp(-r/λ)
    p = np.polyfit(r, np.log(n_sol), 1)
    slope = p[0]
    lambda_num = -1.0 / slope

    # theory
    cs = np.sqrt(T_sol.mean() / cfg.m_i)
    Lpar = L_sol.mean()
    lambda_th = np.sqrt(cfg.D_perp * Lpar / cs)

    print(
        f"[scaling] Measured λ_n={lambda_num:.3e}, "
        f"theory λ_th={lambda_th:.3e}, cs={cs:.3e}, Lpar={Lpar:.3e}"
    )

    return float(lambda_num), float(lambda_th)


def make_sol_width_scaling_figure(
    D_values: np.ndarray,
    lambda_num: np.ndarray,
    lambda_th: np.ndarray,
    outdir: str = "figures",
) -> None:
    """
    Plot SOL scrape-off width scaling vs D_perp:

      • λ_num vs λ_th (1:1 line).
      • λ_num / λ_th vs D_perp^{1/2}.
    """
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # panel (a): λ_num vs λ_th
    ax = axes[0]
    ax.loglog(lambda_th, lambda_num, "o")
    lam_min = min(lambda_th.min(), lambda_num.min()) * 0.7
    lam_max = max(lambda_th.max(), lambda_num.max()) * 1.3
    ax.loglog([lam_min, lam_max], [lam_min, lam_max], "k--", label="1:1")
    ax.set_xlabel(r"$\lambda_{\mathrm{th}}$")
    ax.set_ylabel(r"$\lambda_{\mathrm{3D}}$")
    ax.set_title(r"Scrape-off width: 3D vs 1D theory")
    ax.legend()
    ax.grid(True, which="both")

    # panel (b): ratio vs sqrt(D)
    ax = axes[1]
    ax.plot(np.sqrt(D_values), lambda_num / lambda_th, "o-")
    ax.set_xlabel(r"$D_\perp^{1/2}$")
    ax.set_ylabel(r"$\lambda_{\mathrm{3D}} / \lambda_{\mathrm{th}}$")
    ax.set_title(r"Scaling of $\lambda_n$ with $D_\perp$")
    ax.grid(True)

    fig.tight_layout()
    fname = os.path.join(outdir, "sol_width_scaling.png")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"[fig] Saved SOL-width scaling figure to '{fname}'.")
