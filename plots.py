import os
from typing import Callable, Tuple 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from jax import jit, vmap
import jax.numpy as jnp
Array = jnp.ndarray

# ---------------------------------------------------------------------------
# Plotting: publication-ready figures
# ---------------------------------------------------------------------------

def build_cartesian_grid(cfg) -> Tuple[Array, Array, Array, Array]:
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
    cfg,
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
    cfg,
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
                         cfg,
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

