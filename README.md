# Chip-to-radiator heat budget for orbital AI compute

Three Python scripts behind the note "Chip-to-Radiator Heat Budget for Orbital AI Compute"
(Prachet Poddar, 31 October 2026). They estimate where the temperature budget goes between a
700 W accelerator die and a radiator panel in orbit, for one chip on an isothermal panel. The
conduction totals are first-order lower bounds for the stated materials and assumed geometry.
The hot-spot peaks rest on placeholder power maps and are not bounds in either direction. The
scripts describe no company's actual hardware.

## Files

| File | What it does |
| --- | --- |
| `thermal_budget.py` | Radiator area against the conduction drop, plus the interface layers (TIM and bonded joint). Standard library only. |
| `spreading_stage.py` | Spreading from the die to the joint footprint: eigenfunction series, cross-checked by an independent finite-volume solve. |
| `hotspot_silicon.py` | Adds the silicon die, a local die-attach resistance and a two-zone hot spot. |
| `runs/*.txt` | Output of each script, run on 31 October 2026. |
| `requirements.txt` | Versions used for those runs. |

## Run

```
pip install -r requirements.txt
python thermal_budget.py
python spreading_stage.py
python hotspot_silicon.py
```

Run them from this folder, because the second and third scripts import the first. Each script
runs its self-checks first and prints tables only if every check passes: 31, 41 and 37 checks.

## Inputs

Every input is tagged in the code as PUBLISHED, with its source, or ASSUMED. The hot-spot sizes
and power shares are placeholders: no public power map of any chip was used. The joint
conductance is a patent design value whose reference area is not confirmed.

## Where verification stops

The checks cover internal consistency, units, closed-form limits, energy conservation, grid
convergence (one case), and four stored reference values from a solver a separate reviewer wrote
from the problem statement. That solver is not included, so those four values cannot be
regenerated from this package.
They say nothing about a real, square, non-uniformly heated die on a shared panel. Table rows are
not tested one by one, only the functions they use.

## Environment used

Python 3.13.16, numpy 2.5.3, scipy 1.18.1.

## Licence

MIT. See `LICENSE`.
