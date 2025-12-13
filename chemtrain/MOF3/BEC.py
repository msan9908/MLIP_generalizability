import jax
import jax.numpy as jnp

# NOTE non pbc was not tested !!!!!
def _compute_pol_nonpbc(r_now, q_now):
    """
    Non-periodic polarization:
        P = sum_i q_i * r_i
    r_now: (n_atoms_b, 3)
    q_now: (n_atoms_b, 1) or (n_atoms_b,)
    """
    if q_now.ndim == 1:
        q_now = q_now[:, None]
    # scalar charge per atom -> broadcast to (N, 3)
    q_scalar = q_now[:, 0:1]
    pol = jnp.sum(q_scalar * r_now, axis=0)  # (3,)
    phase = jnp.ones_like(r_now, dtype=jnp.complex64)  # (N_b, 3)
    return pol, phase


def _compute_pol_pbc(r_now, q_now, box_now):
    """
    Periodic polarization with Berry-phase-like formula.

    r_now: (n_atoms_b, 3) (real-space)
    q_now: (n_atoms_b, 1) or (n_atoms_b,)
    box_now: (3, 3)
    """
    if q_now.ndim == 1:
        q_now = q_now[:, None]

    inv_box = jnp.linalg.inv(box_now)
    r_frac = r_now @ inv_box          # (N_b, 3) frac coord
    phase = jnp.exp(1j * 2.0 * jnp.pi * r_frac)  # (N_b, 3)

    q_scalar = q_now[:, 0:1]          # (N_b, 1)
    S = jnp.sum(q_scalar * phase, axis=0)  # (3,)
    pref = 1.0 / (1j * 2.0 * jnp.pi)

    pol = box_now @ S[:, None] * pref   # (3,1)
    pol = pol[:, 0]                     # (3,)
    return pol, phase


def _bec_single_batch(r_now,
                      q_now,
                      box_now,
                      remove_mean=True,
                      epsilon_factor=1.0,
                      output_index=None):
    """
    Compute BEC for a single configuration (one batch index).

    r_now:   (N_b, 3)
    q_now:   (N_b,) or (N_b,1)
    box_now: (3,3) or None
    return:
        if output_index is None: (N_b, 3, 3)
        else:                    (N_b, 3)
    """
    if q_now.ndim == 1:
        q_now = q_now[:, None]

    norm_factor = jnp.sqrt(epsilon_factor)

    def pol_phase_fn(r_arg):
        # r_arg: (N_b, 3)
        qq = q_now
        if remove_mean: # remove mean charge, irrelevant for celli (or at least neutral molecules)
            qq = qq - jnp.mean(qq, axis=0, keepdims=True)

        if box_now is None or jnp.abs(jnp.linalg.det(box_now)) < 1e-6:
            pol, phase = _compute_pol_nonpbc(r_arg, qq)
        else:
            pol, phase = _compute_pol_pbc(r_arg, qq, box_now)

        if output_index is not None:
            pol = pol[output_index]         
            phase = phase[:, output_index]  
        return pol * norm_factor, phase

    # Evaluate at current positions
    pol0, phase0 = pol_phase_fn(r_now)

    if output_index is None:
        # pol0: (3,)
        def pol_only(r_arg):
            p, _ = pol_phase_fn(r_arg)
            return p  # (3,)

        # jac has shape (3, N_b, 3): dP_a / dR_i_beta
        jac = jax.jacrev(pol_only)(r_now)
        bec_complex = jnp.transpose(jac, (1, 2, 0))  # (N_b, 3, 3)

        # phase0: (N_b, 3) -> (N_b, 1, 3) for broadcasting
        phase_exp = jnp.expand_dims(jnp.conj(phase0), axis=1)
        bec = jnp.real(bec_complex * phase_exp)      # (N_b, 3, 3)
        return bec
    else:
        # pol0: scalar
        def pol_only(r_arg):
            p, _ = pol_phase_fn(r_arg)
            return p  # scalar

        # jac: (N_b, 3)
        jac = jax.jacrev(pol_only)(r_now)
        bec_complex = jac  # (N_b, 3)

        # phase0: (N_b,) -> (N_b,1)
        phase_exp = jnp.conj(phase0)[:, None]
        bec = jnp.real(bec_complex * phase_exp)  # (N_b,3)
        return bec


def compute_bec_from_charges(q,
                             r,
                             cell=None,
                             batch=None,
                             remove_mean=True,
                             epsilon_factor=1.0,
                             output_index=None):
    """
    compute BEC tensor from charges and positions 

    q:     (N,) or (N,1)   charges
    r:     (N,3)           positions in real space
    cell:  (3,3) or (B,3,3) or None
    batch: (N,) ints assigning each atom to a batch index [0..B-1], or None.

    Returns:
        if output_index is None:
            bec: (N, 3, 3)
        else:
            bec: (N, 3)
    """
    n_atoms = r.shape[0]

    if q.ndim == 1:
        q = q[:, None]

    if batch is None:
        batch = jnp.zeros(n_atoms, dtype=jnp.int32)

    unique_batches = jnp.unique(batch)

    # Normalize cell to (B,3,3) 
    if cell is not None:
        cell = jnp.array(cell)
        if cell.ndim == 2:
            cell = cell[None, :, :]   # single box for all batches

    
    if output_index is None:
        bec_all = jnp.zeros((n_atoms, 3, 3), dtype=jnp.float32)
    else:
        bec_all = jnp.zeros((n_atoms, 3), dtype=jnp.float32)

    
    # TODO      rework it with lax.scan / vmap and static batch sizes.
    for b in list(unique_batches):
        b_val = int(b)
        mask = (batch == b_val)
        r_now = r[mask]
        q_now = q[mask]

        box_now = None
        if cell is not None:
            box_now = cell[b_val]

        bec_b = _bec_single_batch(
            r_now,
            q_now,
            box_now,
            remove_mean=remove_mean,
            epsilon_factor=epsilon_factor,
            output_index=output_index,
        )

        bec_all = bec_all.at[mask].set(bec_b)

    return bec_all




def polarization_nonpbc(r, q):
    """
    r: (N,3)
    q: (N,) or (N,1)
    returns: polarization vector (3,)
    """
    if q.ndim == 1:
        q = q[:, None]    # make (N,1)
    return jnp.sum(q * r, axis=0)  # (3,)


def compute_bec_single_nonpbc(r, q, remove_mean=True, epsilon_factor=1.0):
    """
    Compute Born effective charge tensor for *one molecule / conformation*.
    
    r: positions (N,3)
    q: charges (N,) or (N,1)
    
    returns:
        bec: (N, 3, 3)
    """
    if q.ndim == 1:
        q = q[:, None]

    if remove_mean:
        q = q - jnp.mean(q, axis=0, keepdims=True)

    norm_factor = jnp.sqrt(epsilon_factor)

    # polarization as function of r (needed for jacobian)
    def pol_fn(r_input):
        return norm_factor * polarization_nonpbc(r_input, q)   # (3,)

    # jac: dP_a / dR_i,beta   shape = (3, N, 3)
    jac = jax.jacrev(pol_fn)(r)   # (3, N, 3)

    # reorder to (N,3,3)
    bec = jnp.transpose(jac, (1, 2, 0))
    return bec

def _pol_nonpbc(r, q):
    """Non-periodic polarization: P = sum_i q_i r_i (real)."""
    if q.ndim == 1:
        q = q[:, None]
    q_scalar = q[:, 0:1]
    return jnp.sum(q_scalar * r, axis=0)  # (3,) real


def _pol_pbc(r, q, box):
    """
    Berry-phase-style polarization under PBC.

    r:   (N,3)
    q:   (N,) or (N,1)
    box: (3,3)
    returns:
        P (complex, shape (3,))
        phase (complex, shape (N,3))
    """
    if q.ndim == 1:
        q = q[:, None]

    inv_box = jnp.linalg.inv(box)
    r_frac = r @ inv_box                     # (N,3), real
    phase = jnp.exp(1j * 2.0 * jnp.pi * r_frac)  # (N,3), complex64

    q_scalar = q[:, 0:1]                    # (N,1)
    S = jnp.sum(q_scalar * phase, axis=0)   # (3,), complex

    pref = 1.0 / (1j * 2.0 * jnp.pi)        # complex
    P = box @ (S[:, None] * pref)           # (3,1), complex
    return P[:, 0], phase                   # (3,), (N,3)


def compute_bec_single_pbc(q, r, cell=None, remove_mean=False, epsilon_factor=1.0):
    """
    Compute Born Effective Charge tensor for a *single configuration*
    (molecule or periodic cell).

    Parameters
    ----------
    q : array, shape (N,) or (N,1)
        Atomic charges.
    r : array, shape (N,3) or flat (3*N,)
        Atomic positions in real space.
    cell : None or (3,3)
        Simulation cell. If None or nearly singular det, treated as non-PBC.
    remove_mean : bool
        Subtract mean charge before polarization.
    epsilon_factor : float
        Overall scaling factor (epsilon_infty).

    Returns
    -------
    bec : array, shape (N, 3, 3)
        Born effective charge tensor:
        bec[i, alpha, beta] = dP_alpha / dR_{i,beta}
    """
    q = jnp.asarray(q)
    r = jnp.asarray(r)

    # ---- Ensure positions shape: (N,3) ----
    if r.ndim == 1:
        if r.size % 3 != 0:
            raise ValueError(f"Flattened r has size {r.size}, not divisible by 3.")
        r = r.reshape(-1, 3)
    elif r.ndim == 2 and r.shape[-1] != 3:
        if r.size % 3 != 0:
            raise ValueError(f"r total size {r.size} not divisible by 3.")
        r = r.reshape(-1, 3)

    N = r.shape[0]

    # ---- Ensure charges shape: (N,1) ----
    if q.ndim == 1:
        q = q.reshape(N, 1)
    else:
        q = q.reshape(N, -1)
        assert q.shape[1] == 1, f"Expected scalar charge per atom, got {q.shape}"

    if remove_mean:
        q = q - jnp.mean(q, axis=0, keepdims=True)

    nf = jnp.sqrt(epsilon_factor)


    # POLARIZATION (REAL output)

    def pol_fn(r_input):
        """Return real-valued polarization for jacrev."""
        if cell is None:
            # Non-PBC: compute real P, then cast to complex for consistency
            P_real = _pol_nonpbc(r_input, q)                  
            P_cplx = jnp.asarray(P_real, dtype=jnp.complex64) # complex64[3]
        else:
            det = jnp.abs(jnp.linalg.det(cell))

            def true_branch(_):
                P_complex, _ = _pol_pbc(r_input, q, cell)     # complex
                return jnp.asarray(P_complex, dtype=jnp.complex64)

            def false_branch(_):
                P_real = _pol_nonpbc(r_input, q)              # float
                return jnp.asarray(P_real, dtype=jnp.complex64)

            P_cplx = jax.lax.cond(det > 1e-6, true_branch, false_branch, operand=None)

        # jacrev requires real output → take real part here
        return jnp.real(nf * P_cplx)                         

    # Jacobian dP/dR: shape (3, N, 3)
    jac = jax.jacrev(pol_fn)(r)

    # Reorder: (N,3,3)
    bec = jnp.transpose(jac, (1, 2, 0))


    # DEPHASING (only if PBC)

    if cell is not None:
        det = jnp.abs(jnp.linalg.det(cell))

        def true_branch(_):
            P_complex, phase = _pol_pbc(r, q, cell)
            P_complex = jnp.asarray(P_complex, dtype=jnp.complex64)
            phase = jnp.asarray(phase, dtype=jnp.complex64)
            return P_complex, phase

        def false_branch(_):
            P_real = _pol_nonpbc(r, q)                        # float
            P_complex = jnp.asarray(P_real, dtype=jnp.complex64)
            phase = jnp.ones_like(r, dtype=jnp.complex64)
            return P_complex, phase

        P_cplx, phase = jax.lax.cond(det > 1e-6, true_branch, false_branch, operand=None)

        # Dephase: phase: (N,3) → (N,1,3)
        phase_exp = jnp.expand_dims(jnp.conj(phase), axis=1)
        bec = jnp.real(bec * phase_exp)

    return bec
def compute_bec_masked(q, r, box, mask):
    """
    q:    (N,) or (N,1)
    r:    (N,3)
    box:  (3,3)
    mask: (N,) or (N,1) with 1=real atom, 0=padding
    """
    # Ensure correct shapes
    q = jnp.asarray(q)
    r = jnp.asarray(r)
    mask = jnp.asarray(mask)

    # reshape to (N,1)
    if q.ndim == 1:
        q = q[:, None]       # (N,1)
    else:
        q = q.reshape(-1, 1)

    if mask.ndim == 1:
        mask = mask[:, None] # (N,1)
    else:
        mask = mask.reshape(-1, 1)

    # Apply mask correctly
    q = q * mask             # (N,1)
    r = r * mask             # (N,3)  * (N,1) 

    return compute_bec_single_pbc(q, r, box)
