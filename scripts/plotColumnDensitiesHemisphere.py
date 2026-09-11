import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.colors import Normalize, LogNorm
from cmcrameri import cm
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{gensymb}"
print(matplotlib.rcParams.keys())
import matplotlib.patheffects as pe

def add_curved_label_chunks(
    ax,
    chunks,
    *,
    t_values,
    offset=0.08,
    color="white",
    fontsize=11,
    path_effects=None,
):
    """
    Draw a label along the upper half-Mollweide boundary using a few chunks.
    Each chunk is placed at its own tangent angle, kept upright.
    """

    for text, t in zip(chunks, t_values):
        # Boundary point
        x = 2 * np.sqrt(2) * np.cos(t)
        y = np.sqrt(2) * np.sin(t)

        # Tangent
        dxdt = -2 * np.sqrt(2) * np.sin(t)
        dydt =  np.sqrt(2) * np.cos(t)
        angle = np.degrees(np.arctan2(dydt, dxdt))

        # Keep text upright
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        # Inward normal of ellipse
        nx = x / (2 * np.sqrt(2))**2
        ny = y / (np.sqrt(2))**2
        norm = np.hypot(nx, ny)
        nx /= norm
        ny /= norm

        # Move inward
        x -= offset * nx
        y -= offset * ny

        txt = ax.text(
            x, y, text,
            color=color,
            fontsize=fontsize,
            ha="center",
            va="center",
            rotation=angle,
            rotation_mode="anchor",
            clip_on=False,
            zorder=10,
        )

        if path_effects is not None:
            txt.set_path_effects(path_effects)

def mollweide_forward(lon, lat):
    """
    Forward Mollweide projection.

    Parameters
    ----------
    lon, lat : arrays in radians

    Returns
    -------
    x, y : projected coordinates
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # Solve 2θ + sin(2θ) = π sin(lat)
    theta = lat.copy()

    # Good special-case handling near the pole
    pole = np.isclose(np.abs(lat), np.pi/2)
    theta[pole] = np.sign(lat[pole]) * np.pi/2

    not_pole = ~pole
    for _ in range(12):
        th = theta[not_pole]
        ph = lat[not_pole]
        f = 2 * th + np.sin(2 * th) - np.pi * np.sin(ph)
        fp = 2 + 2 * np.cos(2 * th)
        theta[not_pole] -= f / fp

    x = 2 * np.sqrt(2) / np.pi * lon * np.cos(theta)
    y = np.sqrt(2) * np.sin(theta)
    return x, y


def edges_from_centers(x, periodic=False, period=None):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("Input must be a 1D array with at least 2 elements.")

    if periodic:
        if period is None:
            raise ValueError("Must provide period for periodic grid.")
        dx = np.diff(x)
        dx_wrap = (x[0] + period) - x[-1]
        edges = np.empty(len(x) + 1)
        edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        edges[0] = x[0] - 0.5 * dx_wrap
        edges[-1] = x[-1] + 0.5 * dx_wrap
        return edges

    dx = np.diff(x)
    edges = np.empty(len(x) + 1)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges


def make_upper_half_mollweide_boundary(n=800):
    """
    Returns a Path for the upper half of the Mollweide ellipse,
    closed along the equator.
    """
    x_max = 2 * np.sqrt(2)
    t = np.linspace(np.pi, 0, n)  # left -> right across top half of ellipse
    x = x_max * np.cos(t)
    y = np.sqrt(2) * np.sin(t)

    verts = np.column_stack([x, y])

    # Close along equator back to start
    verts = np.vstack([
        verts,
        [x_max, 0.0],
        [-x_max, 0.0],
        verts[0]
    ])

    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def plot_half_mollweide_compact(
    az_deg,
    alt_deg,
    values,
    *,
    title="Half-Mollweide",
    cmap="viridis",
    cbar_label="Value",
    central_az_deg=180,
    logscale=False,
    vmin=None,
    vmax=None,
    contours=None,
    figsize=(10, 3.8),
):
    """
    Compact northern-hemisphere-only Mollweide plot.

    Parameters
    ----------
    az_deg : 1D array
        Azimuth centers in degrees, typically spanning 0..360.
    alt_deg : 1D array
        Altitude centers in degrees, typically spanning 0..90.
    values : 2D array
        Shape (len(alt_deg), len(az_deg)).
    central_az_deg : float
        Azimuth shown at the center of the plot.
    contours : sequence or None
        Optional contour levels in data units.
    """
    az_deg = np.asarray(az_deg, dtype=float)
    alt_deg = np.asarray(alt_deg, dtype=float)
    values = np.asarray(values, dtype=float)

    if values.shape != (len(alt_deg), len(az_deg)):
        raise ValueError(
            f"values must have shape ({len(alt_deg)}, {len(az_deg)}), "
            f"got {values.shape}"
        )

    # Use a non-duplicated azimuth grid if both 0 and 360 are present
    if np.isclose((az_deg[-1] - az_deg[0]) % 360, 0) and len(az_deg) > 2:
        if np.isclose(az_deg[-1], az_deg[0] + 360) or np.isclose(az_deg[-1], 360):
            az_deg = az_deg[:-1]
            values = values[:, :-1]

    az_edges_deg = edges_from_centers(az_deg, periodic=True, period=360.0)
    alt_edges_deg = np.clip(edges_from_centers(alt_deg), 0.0, 90.0)

    # Shift azimuth so chosen center sits at x = 0
    lon_edges_deg = (az_edges_deg - central_az_deg + 180.0) % 360.0 - 180.0
    lat_edges_deg = alt_edges_deg  # altitude becomes latitude in upper hemisphere

    Lon_e, Lat_e = np.meshgrid(np.deg2rad(lon_edges_deg), np.deg2rad(lat_edges_deg))
    X_e, Y_e = mollweide_forward(Lon_e, Lat_e)

    # Centers for contours
    lon_cent_deg = (az_deg - central_az_deg + 180.0) % 360.0 - 180.0
    lat_cent_deg = alt_deg
    Lon_c, Lat_c = np.meshgrid(np.deg2rad(lon_cent_deg), np.deg2rad(lat_cent_deg))
    X_c, Y_c = mollweide_forward(Lon_c, Lat_c)

    if logscale:
        if np.any(values <= 0):
            raise ValueError("All values must be > 0 for log scaling.")
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)

    mesh = ax.pcolormesh(
        X_e, Y_e, values,
        shading="auto",
        cmap=cmap,
        norm=norm,
        edgecolors='none',
        linewidth=0,
        rasterized=True
    )

    # Clip to upper half-Mollweide boundary
    boundary = make_upper_half_mollweide_boundary()
    clip_patch = PathPatch(boundary, transform=ax.transData, facecolor="none")
    mesh.set_clip_path(clip_patch)

    # Draw outline
    outline = PathPatch(
        boundary,
        transform=ax.transData,
        facecolor=cm.lipari(norm(values[0,0])),
        edgecolor="black",
        lw=1.1,
        zorder=0
    )
    ax.add_patch(outline)
    outline = PathPatch(
        boundary,
        transform=ax.transData,
        facecolor="none",
        edgecolor="black",
        lw=1.1,
        zorder=5
    )
    ax.add_patch(outline)

    cs = ax.contour(
        X_c, Y_c, values,
        levels=contours,
        colors="white",
        linewidths=0.8,
    )

    cs.set_clip_path(clip_patch)

    # Equator line
    x_eq = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), 600)
    ax.plot(x_eq, np.zeros_like(x_eq), color="black", lw=1.0, zorder=6)

    # Latitude guide curves (altitudes)
    guide_lats = [15, 30, 45, 60, 75]
    offsetXs   = [.11, .12, .14, .17, .20]
    lon_line = np.linspace(-np.pi, np.pi, 800)
    for latd, offsetX in zip(guide_lats, offsetXs):
        lat_line = np.deg2rad(np.full_like(lon_line, latd))
        xg, yg = mollweide_forward(lon_line, lat_line)
        ax.plot(xg, yg, color="0.75", lw=0.7, zorder=1, alpha = .05)

        # label near left edge
        xlab, ylab = mollweide_forward(np.array([-np.pi + 0.06]), np.array([np.deg2rad(latd)]))
        print(xlab[0])
        ax.text(xlab[0] - offsetX, ylab[0], f"${latd}$", ha="right", va="center", fontsize=9)

    # Meridian guide curves
    meridians = np.arange(-150, 180, 30)
    lat_line = np.linspace(0, np.pi / 2, 400)
    for md in meridians:
        lon_line = np.deg2rad(np.full_like(lat_line, md))
        xg, yg = mollweide_forward(lon_line, lat_line)
        ax.plot(xg, yg, color="0.85", lw=0.6, zorder=1, alpha = .05)

    # Azimuth labels along equator
    label_meridians = np.arange(-180, 180 + 30, 30)
    for md in label_meridians:
        az_label = int(md + central_az_deg)
        xt, yt = mollweide_forward(np.array([np.deg2rad(md)]), np.array([0.0]))
        ax.text(xt[0], yt[0] - 0.06, f"${az_label}$", ha="center", va="top", fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlim(-2 * np.sqrt(2) * 1.02, 2 * np.sqrt(2) * 1.02)
    ax.set_ylim(-0.12, np.sqrt(2) * 1.02)
    ax.axis("off")

    add_curved_label_chunks(
        ax,
        chunks=[
            "altitude",
            r"$\theta\ (^\circ)$"
        ],
        #t_values=[0.8*np.pi, 0.73*np.pi],
        t_values=[0.75 * np.pi, 0.69 * np.pi],
        offset=0.10,
        color="white",
        fontsize=11,
        path_effects=[pe.withStroke(linewidth=1.5, foreground=".3")],
    )

    imax = np.unravel_index(np.argmax(values), values.shape)
    alt_peak = alt_deg[imax[0]]
    az_peak = az_deg[imax[1]]

    x_peak, y_peak = mollweide_forward(
        np.deg2rad(az_peak - central_az_deg),
        np.deg2rad(alt_peak)
    )

    ax.scatter(
        x_peak, y_peak,
        marker="*",
        s=50,
        color="cornflowerblue",
        edgecolors=".3",
        linewidths=1.,
        zorder=20
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        pad=0.12,
        fraction=0.04,
        aspect=72,
    )
    cbar.set_label(cbar_label)

    ax.text(
        0.5, +0.18,
        r"azimuth $\varphi\ (^\circ)$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="white",
        path_effects=[pe.withStroke(linewidth=1.5, foreground=".3")]
    )

    # Draw in the top-right corner of the axes
    ax.text(
        1.01,
        1.02,
        r"\textbf{host galaxy:}"+"\n" + rf"$\alpha = {ra:.1f}\degree,\ \delta = {dec:.1f}\degree$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="0.3",
        fontsize=10,
        zorder=30,
    )

    #plt.tight_layout()
    return fig, ax


# Example grid
azimuths  = np.linspace(0, 360, 360, endpoint = False)
altitudes = np.linspace(0, 90, 91)
data      = np.load("/Users/martijnoei/Library/CloudStorage/Dropbox/Martijn/Caltech/Caltech Connection/ten_excels_1.26.26_Mpc/Mpc_column_densities_all_r.npy")

import pandas as pd
xlsx_path = "/Users/martijnoei/Library/CloudStorage/Dropbox/Martijn/Caltech/Caltech Connection/ten_excels_1.26.26_Mpc/Mpc_filament_pa_exact_1.xlsx"
# Read the table
df = pd.read_excel(xlsx_path)
# Adjust these column names if needed


#indexJetSystem = 137

for indexJetSystem in range(data.shape[0]):
    ra  = float(df.loc[indexJetSystem, "right_ascension (deg)"])
    dec = float(df.loc[indexJetSystem, "declination (deg)"])
    fig, ax = plot_half_mollweide_compact(
        azimuths,
        altitudes,
        data[indexJetSystem],
        cmap=cm.lipari,
        cbar_label=r"Cosmic Web column density $\sigma_\mathrm{CW}\ (\mathrm{g\ m^{-2}})$",
        central_az_deg=180,
        contours=[1.5e20, 2.0e20, 2.5e20, 3.0e20],
        figsize=(6, 2.5),
    )
    plt.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.2)
    plt.savefig(f"/Users/martijnoei/Library/CloudStorage/Dropbox/Martijn/Caltech/Caltech Connection/columnDensitiesHemisphere_{indexJetSystem}.pdf", dpi = 1000)
    plt.close()