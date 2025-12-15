# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom functions simplifying the handling of fractional coordinates."""
from typing import Union, Tuple, Callable

from jax_md import space, util
import jax.numpy as jnp
from jax import vmap

Box = Union[float, util.Array]


def _rectangular_boxtensor(box: Box) -> Box:
    """Transforms a 1-dimensional box to a 2D box tensor."""
    spatial_dim = box.shape[0]
    return jnp.eye(spatial_dim).at[jnp.diag_indices(spatial_dim)].set(box)


def init_fractional_coordinates(box: Box) -> Tuple[Box, Callable]:
    """Returns a 2D box tensor and a scale function that projects positions
    within a box in real space to the unit-hypercube as required by fractional
    coordinates.

    Args:
        box: A 1 or 2-dimensional box

    Returns:
        A tuple (box, scale_fn) of a 2D box tensor and a scale_fn that scales
        positions in real-space to the unit hypercube.
    """
    if box.ndim != 2:
        box = _rectangular_boxtensor(box)

    def scale_fn(positions, **kwargs):
        _box = kwargs.get('box', box)
        if _box.ndim != 2:
            _box = _rectangular_boxtensor(_box)
        inv_box = jnp.linalg.inv(_box)
        return jnp.dot(inv_box, positions.T).T

    return box, scale_fn


def nonperiodic_general(box: Box = None,
                        fractional_coordinates: bool = True,
                        wrapped: bool = True) -> space.Space:
    """Nonperiodic boundary conditions on a parallelepiped."""
    del wrapped
    
    if box is not None:
        inv_box = space.inverse(box)
    else:
        inv_box = None

    def displacement_fn(Ra, Rb, perturbation=None, **kwargs):
        _box, _inv_box = box, inv_box

        if 'box' in kwargs:
            assert fractional_coordinates, (
                "Coordinates must be fractional to specify a new box."
            )
            _box = kwargs['box']

        if 'new_box' in kwargs:
            assert fractional_coordinates, (
                "Coordinates must be fractional to specify a new box."
            )
            _box = kwargs['new_box']

        dR = space.pairwise_displacement(Ra, Rb)
        if fractional_coordinates:
            assert _box is not None, (
                "Box must be provided for fractional coordinates."
            )
            dR = space.transform(_box, dR)

        if perturbation is not None:
            raise NotImplementedError(
                "Perturbation not supported for nonperiodic_general.")

        return dR

    def shift_fn(R, dR, **kwargs):
        return R + dR

    return displacement_fn, shift_fn
