
import jax.numpy as jnp

import logging

import chex
from optax import Schedule

def polynomial_schedule(
    init_value: chex.Scalar,
    end_value: chex.Scalar,
    power: chex.Scalar,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
  r"""Constructs a schedule with polynomial transition from init to end value.

  >>> from chemtrain.learn import schedules
  >>>
  >>> sched = polynomial_schedule(0.1, 0.01, 0.33, 10, 1)
  >>>
  >>> sched(0)
  0.1
  >>> sched(1)
  0.1
  >>> sched(11)
  0.01
  >>> sched(11)
  0.01

  Args:
    init_value: initial value for the scalar to be annealed.
    end_value: end value of the scalar to be annealed.
    power: the power of the polynomial used to transition from init to end.
    transition_steps: number of steps over which annealing takes place.
    transition_begin: must be positive. After how many steps to start annealing
      (before this many steps the scalar value is held fixed at ``init_value``).

  Returns:
    schedule
      A function that maps step counts to values.

  """
  if transition_steps <= 0:
    logging.info(
        'A polynomial schedule was set with a non-positive `transition_steps` '
        'value; this results in a constant schedule with value `init_value`.'
    )
    return lambda count: init_value

  if transition_begin < 0:
    logging.info(
        'A polynomial schedule was set with a negative `transition_begin` '
        'value; this will result in `transition_begin` falling back to `0`.'
    )
    transition_begin = 0

  def schedule(s):
    s = jnp.clip(s - transition_begin, 0, transition_steps)

    b = transition_steps / (jnp.pow(init_value / end_value, 1 / power) - 1)
    a = init_value * jnp.pow(b, power)

    return a * jnp.pow(b + s, -power)

  return schedule