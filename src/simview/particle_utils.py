import numpy as np

MEC2_EV = 510998.95


def kinetic_energy_eV(part, species=1):
    """
    Relativistic particle kinetic energy from beta-gamma components.
    """
    mask = part.species == species
    bg2 = (
        part.bgx[mask]**2
        + part.bgy[mask]**2
        + part.bgz[mask]**2
    )
    gamma = np.sqrt(1.0 + bg2)
    return (gamma - 1.0) * MEC2_EV


def density(part, species=1):
    """
    Charge-weighted particle density for a given species.
    """
    mask = part.species == species
    weights = np.abs(part.q[mask])/1.602e-13  # Convert charge to number of particles
    return np.sum(weights)


def diagnostics(part, species=1):
    """
    Calculate diagnostic quantities for the particle distribution.
    """
    mask = part.species == species

    weights = np.abs(part.q[mask])/1.602e-13  # Convert charge to number of particles
    energy = kinetic_energy_eV(part, species=species)

    n_sp = density(part, species=species)
    total_energy = np.sum(weights * energy)
    mean_energy = total_energy / n_sp

    diagnostics = {
        "weights": weights,
        "total_energy": total_energy,
        "mean_energy": mean_energy,
        "density": n_sp,
        "n_parts": len(weights),
    }

    return diagnostics


def eedf(part, edges, species=1, normalize=True):
    """
    Charge-weighted electron energy distribution.

    Returns the differential distribution evaluated over energy bins.
    """
    mask = part.species == species

    energy = kinetic_energy_eV(part, species=species)
    weight = np.abs(part.q[mask])/1.602e-13  # Convert charge to number of particles

    hist, _ = np.histogram(
        energy,
        bins=edges,
        weights=weight,
    )

    dE = np.diff(edges)

    if normalize:
        total = hist.sum()
        if total > 0:
            hist = hist / total

    return hist / dE