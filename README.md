# Iberian_blackout
A simplified replication of the April 28, 2025 Iberian Peninsula cascade using the IEEE 39-bus power system on ANDES.
## Requirements
| Requirement | Notes |
|-------------|-------|
| [ANDES](https://github.com/cuihantao/andes) ≥ 1.6 | Provides the dynamic simulation engine. |
| NumPy, SciPy, pandas, matplotlib | Installed automatically when you set up the plotting environment. |

## Running the Simulation
1. Make sure the ANDES case files are available (they come with the standard ANDES install).
2. Execute the final scenario:

   ```bash
   python final.py
   ```

   This generates the time-domain trajectories and saves the CSV/JSON outputs used by the figures:

   - `iberian_cascade_bus_voltages.csv`
   - `iberian_cascade_collector_voltages.csv`
   - `iberian_cascade_events.json`

## Generating Figures
After `final.py` finishes, simply run:

```bash
python figures.py
```

The script reads the CSV/JSON files above and creates the plots (PDFs in the repo root).
