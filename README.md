Overview

This project investigates anomalous and statistically unusual behaviour in auxiliary channels of the LIGO Hanford Observatory, introducing a multichannel signal phenotyping framework designed to identify, characterise, and test the reproducibility of structured signal behaviour across detector subsystems.

LIGO auxiliary channels provide information about the detector environment, sensing systems, control systems, and instrumental state. These channels can contain transient non-astrophysical signatures that may be relevant to detector characterisation, data-quality investigations, and understanding the behaviour of complex interferometric systems. Rather than treating anomalies as isolated single-channel events, this project investigates their **multichannel phenotype**: the quantitative combination of temporal, spectral, statistical, and cross-channel properties associated with an anomalous event.

The analysis pipeline performs candidate-window detection, temporal and spectral feature extraction, statistical characterisation, dimensionality reduction, cross-channel correlation and lag analysis, and event-level feature comparison. Candidate events are represented using quantitative feature vectors and examined across independent time windows to determine whether apparently unusual signals exhibit reproducible multichannel structure.

The current implementation analyses O4a-era LIGO Hanford data and generates candidate-event summaries, principal-component representations, spectral diagnostics, correlation and null-model distributions, lagged cross-channel relationships, and research visualisations. The pipeline is designed to preserve a clear separation between raw detector data, derived measurements, statistical analyses, and reproducible research outputs.

The broader objective is to establish a generalisable framework for **multichannel anomaly phenotyping** that can be independently reproduced, evaluated against appropriate null models, extended to larger observational datasets, and ultimately used to investigate the instrumental and physical origins of persistent or recurring auxiliary-channel anomalies.

Current results are exploratory. Apparent anomalous structure is not interpreted as evidence of a confirmed physical phenomenon without independent replication, statistical validation, and investigation of instrumental or environmental explanations.
