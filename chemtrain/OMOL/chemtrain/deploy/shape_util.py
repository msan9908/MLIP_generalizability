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

"""Helper functions to infer dynamic shapes."""

from typing import List, Any, Callable


def define_symbols(symbols: str, constraints: List[str] = None):
    """Delays the definition of symbols until all symbols and constraints are known.

    Args:
        symbols: String of symbols to define.
        constraints: List of constraints to apply to the symbols.
    """

    def decorator(f):
        symb = [s.lstrip() for s in symbols.split(',')]

        def apply_fn(**defined_symbols: Any):

            # Pass the defined symbols as positional arguments
            args = [defined_symbols.pop(s) for s in symb]

            return f(*args, **defined_symbols)

        def wrapped(s: List[Any], c: List, apply_fns: List[Callable]):
            assert set(s).isdisjoint(set(symb)), "Symbols already defined"

            s.extend(symb)
            apply_fns.append(apply_fn)

            if constraints is not None:
                c.extend(constraints)

        return wrapped
    return decorator
