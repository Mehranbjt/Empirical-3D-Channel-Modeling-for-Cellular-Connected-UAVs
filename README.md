# Empirical LTE Air-to-Ground (A2G) Channel Measurement Dataset for Cellular-Connected UAVs

This repository provides an empirical measurement dataset collected through extensive aerial UAV drive tests conducted within an operational Long-Term Evolution (LTE) cellular network deployment. The dataset is specifically designed for research in cellular-connected UAVs, air-to-ground (A2G) radio propagation, and machine learning-based 3D channel modelling.

**Dataset Summary**
- Over 95,000 raw measurement samples collected
- Final dataset contains ~11,060 meticulously filtered and validated samples
- Contains key RF metrics:
  - Reference Signal Received Power (RSRP)
  - Reference Signal Received Quality (RSRQ)
  - Received Signal Strength Indicator (RSSI)
  - Path Loss (PL)

- Includes rich spatial metadata:
  - Log-transformed 3D / 2D propagation distances
  - Azimuth and Elevation angles relative to serving antenna
  - UAV altitude metadata

This dataset is suitable for ML-driven A2G channel characterization, RS-based propagation analysis, link adaptation research, and model generalization studies for dense urban UAV-assisted wireless networks.

=================================================================================

**Preprocessing Scripts**

Two Python preprocessing modules are provided in this repository:

- **preprocess_lte_dataset.py**	Cleans raw measurement CSV, normalizes features, removes invalid rows, optionally labels RSRP strength categories.
- **preprocess_uav_lte_geometry.py**	Processes altitude-based LTE logs, computes 2D and 3D distances + optional azimuth/elevation geometry, isolates pcell/detected measurements, optional PCI filtering, merges and deduplicates outputs.

These scripts convert the raw field-logged CSV data into clean, ML-ready datasets suitable for model training pipelines.

=================================================================================

**Machine Learning Model (Triple-Layer Architecture)**

A triple-layer ML model (used in the original research, Empirical 3D Channel Modeling for Cellular-Connected UAVs: A Triple-Layer Machine Learning Approach) can be shared upon request.

To request access, please contact:

haideralobaidy@nahrainuniv.edu.iq
