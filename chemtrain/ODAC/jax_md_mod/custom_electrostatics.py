import functools
from typing import Union, Iterable

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import numpy as onp

import lineax

from jax_md import energy, smap, space, util as md_util, quantity
from jax_md._energy import electrostatics
from jax_md_mod import custom_quantity


def structure_factor(g, R, q=1, mask=None):
    if mask is None:
        mask = jnp.ones(R.shape[0], dtype=bool)

    if isinstance(q, jnp.ndarray):
        q = q[None, :]
    return md_util.high_precision_sum(
        q * jnp.exp(1j * jnp.einsum('id,jd->ij', g, R)) * mask,
        axis=1
    )


def shielded_interaction(dr, charge, alpha, alpha_max=None):
    """Gaussian (shielded) charge interaction."""
    # Safety: Avoid division by zero
    mask = dr > 1e-7
    dr = jnp.where(mask, dr, 1e-7)

    if alpha_max is not None:
        print(f"Apply shielding")
        erfdiff = jsp.special.erf(alpha * dr) - jsp.special.erf(alpha_max * dr)
        pot = mask * charge * erfdiff / dr
    else:
        pot = mask * charge * jsp.special.erf(alpha * dr) / dr

    return pot


def shielded_self(charge, radii):
    """Gaussian (shielded) self-interaction."""
    return jnp.sum(charge * charge / (2 * radii * jnp.sqrt(jnp.pi)))


def core_interaction(charge, chi, idmp):
    """Core interaction."""
    return jnp.sum((chi + charge * idmp / 2) * charge)


def custom_coulomb_recip_ewald(charge,
                               box,
                               alpha: float,
                               grid: Union[int, Iterable[int]],
                               fractional_coordinates=False):
  def energy_fn(position, mask=None, **kwargs):
    n_particles, dim = position.shape

    _box = kwargs.get("box", box)
    if mask is None:
        mask = jnp.ones(n_particles, dtype=bool)

    # Create a grid of reciprocal vectors
    if jnp.isscalar(_box) or jnp.shape(_box) == ():
        _box = jnp.eye(dim) * _box
    elif jnp.ndim(box) == 1:
        _box = jnp.diag(_box)
    else:
        assert jnp.shape(_box) == (dim, dim)

    volume = quantity.volume(dim, _box)
    _invbox = jnp.linalg.inv(_box)

    if fractional_coordinates:
        position = jnp.einsum('ij,nj->ni', _box, position)

    # Non-homogeneous grid dimension
    if isinstance(grid, int):
      _grid = [grid] * dim
    else:
      _grid = grid

    # Create a grid with the specified dimensions but omit all-zero wave vectors
    m = jnp.meshgrid(*(jnp.arange(g) for g in _grid), indexing='ij')
    g = jnp.stack([m[i].ravel() for i in range(dim)], axis=-1)[1:, :]

    # Inverse box gives reciprocal lattice vectors as rows
    g = 2 * jnp.pi * jnp.einsum('ji,nj->ni', _invbox, g)
    g2 = jnp.sum(g ** 2, axis=-1)

    # Compute the structure factors
    S = structure_factor(g, position, charge, mask=mask)
    S2 = jnp.real(jnp.conj(S) * S)

    # Double the sum due to purely positive wave vectors
    pot = jnp.exp(-g2 / (4 * alpha ** 2)) / g2 * S2
    pot = 4 * jnp.pi / volume * jnp.sum(pot)

    return pot
  return energy_fn


def custom_coulomb_recip_pme(charge,
                             box,
                             grid,
                             fractional_coordinates: bool=False,
                             alpha: float=0.34
                             ):
  _ibox = space.inverse(box)
  _grid = grid

  def energy_fn(R, **kwargs):
    q = kwargs.pop('charge', charge)
    _box = kwargs.pop('box', box)
    ibox = space.inverse(kwargs['box']) if 'box' in kwargs else _ibox

    dim = R.shape[-1]
    if isinstance(_grid, int):
        grid_dimensions = [_grid] * dim
    else:
        grid_dimensions = _grid

    grid_dimensions = onp.asarray(grid_dimensions)

    grid = electrostatics.map_charges_to_grid(R, q, ibox, grid_dimensions,
                               fractional_coordinates)
    Fgrid = jnp.fft.fftn(grid)

    mx, my, mz = jnp.meshgrid(*[jnp.fft.fftfreq(g) for g in grid_dimensions])
    if jnp.isscalar(_box):
      m_2 = (mx**2 + my**2 + mz**2) * (grid_dimensions[0] * ibox)**2
      V = _box**dim
    else:
      m = (ibox[None, None, None, 0] * mx[:, :, :, None] * grid_dimensions[0] +
           ibox[None, None, None, 1] * my[:, :, :, None] * grid_dimensions[1] +
           ibox[None, None, None, 2] * mz[:, :, :, None] * grid_dimensions[2])
      m_2 = jnp.sum(m**2, axis=-1)
      V = jnp.linalg.det(_box)
    mask = m_2 != 0

    exp_m = 1 / (2 * jnp.pi * V) * jnp.exp(-jnp.pi**2 * m_2 / alpha**2) / m_2
    return md_util.high_precision_sum(
        mask * exp_m * electrostatics.B(mx, my, mz) * jnp.abs(Fgrid)**2)
  return energy_fn


def shielded_interaction_neighbor_list(displacement_fn, r_onset, r_cutoff, box=None, alpha=4.5, grid=None, method="reciprocal"):
    """Gaussian (shielded) charge interaction."""

    def energy_fn(position, neighbor, charge, radii, chi=None, idmp=None, equilibrate=True, precondition=False, fractional_coordinates=True, **dynamic_kwargs):
        if method == "direct":
            _energy_fn = smap.pair_neighbor_list(
                energy.multiplicative_isotropic_cutoff(
                    shielded_interaction, r_onset, r_cutoff
                ),
                space.metric(displacement_fn),
                charge=(lambda q1, q2: q1 * q2, charge),
                alpha=(lambda s1, s2: 1 / jnp.sqrt(2 * (s1 ** 2 + s2 ** 2)), radii),
            )
            pot = 0.0
        elif method in ["ewald", "pme"]:
            _box = dynamic_kwargs.get("box", box)

            assert _box is not None, "Box must be provided for reciprocal space calculation."

            if method == "ewald":
                recip_fn = lambda pos, charge, **kwargs: custom_coulomb_recip_ewald(charge, _box, alpha, grid=grid, fractional_coordinates=fractional_coordinates)(pos, **kwargs)
            else:
                recip_fn = lambda pos, charge, **kwargs: custom_coulomb_recip_pme(charge, _box, grid=grid, fractional_coordinates=fractional_coordinates, alpha=alpha)(pos, **kwargs)

            _energy_fn = smap.pair_neighbor_list(
                energy.multiplicative_isotropic_cutoff(
                    shielded_interaction, r_onset, r_cutoff
                ),
                space.metric(displacement_fn),
                charge=(lambda q1, q2: q1 * q2, charge),
                alpha=(lambda s1, s2: 1 / jnp.sqrt(2 * (s1 ** 2 + s2 ** 2)), radii),
                alpha_max=alpha
            )

            pot = 0.0
            if not precondition:
                pot += recip_fn(position, charge, **dynamic_kwargs)
                pot -= shielded_self(charge, 1 / (2 * alpha)) # Correct for the self-interaction added in reciprocal space

            # jax.debug.print("Reciprocal energy: {}", pot)
            # jax.debug.print("Reciprocal gradient: {}", jax.grad(recip_fn, argnums=1)(position, charge, **dynamic_kwargs))
            # jax.debug.print("Reciprocal hessian: {}", jax.hessian(recip_fn, argnums=1)(position, charge, **dynamic_kwargs))


        # jax.debug.print("Real space energy: {}", _energy_fn(position, neighbor))
        # jax.debug.print("Self-interaction energy: {}", shielded_self(charge, radii))

        pot += shielded_self(charge, radii)
        pot += _energy_fn(position, neighbor)

        # Add electronegativity and hardness terms
        if chi is not None and idmp is not None:
            if equilibrate:
                print(f"Add core interaction")
                pot += core_interaction(charge, chi, idmp)
            else:
                pot += core_interaction(charge, chi, idmp)

        return pot

    return energy_fn

# TODO: Add alpha interaction parameter
def charge_eq_energy_neighborlist(displacement, r_onset, r_cutoff, interaction="shielded", solver="direct", electrostatics="direct", grid=None, max_local: int = None, alpha=4.5, fractional_coordinates=True, box=None):
    """Charge equilibration energy function."""
    method = solver
    if interaction == "shielded":
        total_energy_fn = shielded_interaction_neighbor_list(displacement, r_onset, r_cutoff, method=electrostatics, grid=grid, alpha=alpha)
    else:
        raise ValueError(f"Unknown interaction {interaction}")

    def energy_fn(position, neighbor, radii=None, chi=None, idmp=None, mask=None, total_charge=None, charge=None, **dynamic_kwargs):
        if mask is None:
            mask = jnp.ones(position.shape[0], dtype=bool)
        if total_charge is None:
            total_charge = 0.0
            print(f"No total charge specified. Total charge will be set to {total_charge}")
        else:
            print(f"Total charge specified: {total_charge}")

        # Evaluate for precomputed charges
        if charge is not None:
            return total_energy_fn(
            position, neighbor, charge=charge, radii=radii, chi=chi, idmp=idmp,
            equilibrate=False, **dynamic_kwargs
        )

        n_particles = mask.size

        if max_local is None:
            charge = jnp.zeros(n_particles)
        else:
            charge = jnp.zeros(max_local)

        if method == "direct":
            if max_local is None:
                # Count number of particles
                A = jnp.zeros((n_particles + 1, n_particles + 1))

                # Set last row (charge neutrality) to mask
                A = A.at[-1, :-1].set(mask)

                # Set row for muliplier to mask
                A = A.at[:-1, -1].set(mask)

                # Set diagonal entries to hessian. As the charges minimize the
                # energy, the gradient of the coloumb interactions depend on the
                # position only explicitly but not through the charges.

                A = A.at[:-1, :-1].set(
                    jax.hessian(total_energy_fn, argnums=2)(
                        position, neighbor, charge, radii,
                        chi, idmp, **dynamic_kwargs
                    )
                )

                # A = A.at[:-1, :-1].set(jnp.diag(~mask) + A[:-1, :-1])

                # jax.debug.print("Coulomb matrix: {}", A)

                # Charge neutrality constraint (for now)
                b = jnp.concatenate((-chi, jnp.full((1,), total_charge))).reshape((-1, 1))

                # Solve the linear system with lagrange multipliers
                charges = jsp.linalg.solve(A, b, assume_a="sym")[:-1, 0]

            else:
                print(f"Consider only {max_local} local atoms")

                A = jnp.zeros((max_local + 1, max_local + 1))

                n_local = jnp.sum(mask) // dynamic_kwargs["reps"]
                local_mask = jnp.arange(max_local) < n_local

                A = A.at[-1, :max_local].set(mask[:max_local] & local_mask)

                A = A.at[:max_local, -1].set(mask[:max_local] & local_mask)

                tile_idx = jnp.mod(jnp.arange(n_particles), n_local)
                def _periodic_hessian(q):
                    charge = q[tile_idx] * (jnp.arange(mask.size) < (n_local * dynamic_kwargs["reps"]))
                    return total_energy_fn(position, neighbor, charge, radii, chi, idmp, **dynamic_kwargs)

                A = A.at[:-1, :-1].set(
                    jax.hessian(_periodic_hessian)(charge)
                )

                A += jnp.diag(jnp.append(~local_mask, 0))

                # Charge neutrality constraint (for now)
                b = jnp.concatenate(
                    (-chi[:max_local] * local_mask * dynamic_kwargs["reps"], jnp.full((1,), total_charge))).reshape((-1, 1))

                # Solve the linear system with lagrange multipliers
                local_charges = jsp.linalg.solve(A, b, assume_a="sym")[:-1, 0]

                charges = jnp.where(
                    jnp.arange(mask.size) < n_local * dynamic_kwargs["reps"],
                    local_charges[tile_idx],
                    0.0
                )

        elif method == "CG":
            # Count number of particles
            A = jnp.zeros((n_particles + 1, n_particles + 1))

            # Set last row (charge neutrality) to mask
            A = A.at[-1, :-1].set(mask)

            # Set row for muliplier to mask
            A = A.at[:-1, -1].set(mask)

            # Set diagonal entries to hessian. As the charges minimize the
            # energy, the gradient of the coloumb interactions depend on the
            # position only explicitly but not through the charges.
            A = A.at[:-1, :-1].set(
                jax.hessian(
                    functools.partial(total_energy_fn, precondition=True),
                    argnums=2
                )(
                    position, neighbor, charge, radii,
                    chi, idmp, **dynamic_kwargs
                )
            )

            # Ideally a sparse approximate inverse
            lup = jsp.linalg.lu_factor(A)
            lup = jax.lax.stop_gradient(lup)

            def linear_operator(x):
                print(f"Shape of x is {x}")
                charge = x[:-1, 0]
                mult = x[-1, 0]

                Jq = jax.grad(total_energy_fn, argnums=2)(
                    position, neighbor, charge,
                    radii, chi, idmp, **dynamic_kwargs
                )

                Ax = (Jq + mult) * mask + (1 - mask) * charge
                return jnp.append(Ax, jnp.sum(mask * charge)).reshape((-1, 1))

            def preconditioner(x):
                return jsp.linalg.lu_solve(lup, x)

            # Initial guess
            x0 = jnp.zeros(n_particles + 1).reshape((-1, 1))
            b = jnp.concatenate((-chi, jnp.full((1,), total_charge))).reshape((-1, 1))

            sol, _ = jsp.sparse.linalg.bicgstab(
                linear_operator, b, x0=x0, tol=1e-8,
                #M=preconditioner
            )
            charges = sol[:-1, 0]

        elif method == "lineax":
            operator = lineax.JacobianLinearOperator(
                lambda x: total_energy_fn(position, neighbor, x[:-1], radii, chi, idmp, **dynamic_kwargs) + x[-1] * jnp.sum(mask * x[:-1]),
            )
            solution = lineax.CG().compute(operator, jnp.concatenate((-chi, jnp.full((1,), total_charge))), {})
            sol, *other = solution

            charges = sol[:-1]

            # raise NotImplementedError("CG method not implemented yet.")
        else:
            raise ValueError(f"Unknown method {method}")

        charges = jnp.where(mask, charges, 0.0)

        qeq_energy = total_energy_fn(
            position, neighbor, charge=charges, radii=radii, chi=chi, idmp=idmp,
            equilibrate=False, **dynamic_kwargs
        )

        # Only include electrostatic energy
        # qeq_energy = total_energy_fn(
        #     position, neighbor, charge=charges, radii=radii, **dynamic_kwargs)
        return qeq_energy, charges

    return energy_fn
