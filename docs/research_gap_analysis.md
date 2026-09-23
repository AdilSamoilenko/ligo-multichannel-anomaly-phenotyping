# Research Gap Analysis

## 1. Executive Summary

This analysis uses findings established in the previous research session and does not introduce new experimental results. The surveyed landscape is dominated by supervised morphology-based glitch classification and heuristic vetoes. The previous summary does not identify a systematic comparison of unsupervised multichannel temporal representations across auxiliary channels for O4-era detector-behaviour characterisation. Candidate gaps C1, C2 and C3 therefore remain open and require cautious validation before any claim of novelty.

## 2. What Has Already Been Done

Gravity Spy has provided a large supervised taxonomy of gravitational-wave glitches using citizen science and convolutional networks. Omicron offers outlier detection for triggers and auxiliary-channel flags. Tools such as hveto and OVL support instrumental vetoes and noise identification. Auxiliary-channel coherence studies have used cross-channel correlation to characterise environmental and instrumental disturbances. Unsupervised anomaly-detection methods, including autoencoders and isolation forests, have flagged unusual segments but have not been systematically benchmarked against auxiliary-channel temporal representations. Machine-learning classifiers have improved known-family identification but remain tied to existing morphology taxonomies. Temporal and transformer models can capture long-range sequential structure, though many remain strain-focused or supervised. Spiking neural network approaches are emerging and may support low-latency detection. DANTE V3 and its O4a result represent the most recent anomaly-detection reference in the established summary, although full method details and validation were not available in that summary.

## 3. Assessment of the Original Hypothesis

The hypothesis that multichannel temporal representations of detector data can reveal previously uncharacterised detector behaviour is plausible but not established by the surveyed materials. Existing supervised systems are optimised for known glitch morphology. Unsupervised methods can flag anomalies but do not by themselves validate new physical or instrumental interpretations. The previous summary does not contain evidence that auxiliary-channel temporal modelling alone uncovers behaviour beyond the strain-based morphology catalogue. The hypothesis therefore remains a research direction rather than a demonstrated result.

## 4. Candidate Research Gaps

C1: Absence of a systematic comparison of unsupervised anomaly detection across joint strain-auxiliary temporal representations for O4-era data. This specific benchmark was not identified in the surveyed literature.

C2: Limited work explicitly models temporal structure in auxiliary channels to characterise detector behaviour beyond glitch classification. Transformer and neuromorphic methods remain mostly strain-focused or at an early stage, and their utility for auxiliary-channel discovery is not established in the previous summary.

C3: Few studies combine coherence and correlation across auxiliary channels with anomaly-discovery frameworks to surface behaviour not captured by morphology-based families. Coherence is commonly used for vetoes rather than exploratory discovery, so this combination is not identified in the surveyed materials.

## 5. Research Direction

A comparative evaluation should test representative auxiliary-channel temporal features against unsupervised anomaly-detection methods, including autoencoders, isolation forest and possibly transformer or spiking models. Coherence-derived features should be included alongside raw temporal features. Results should be compared against known glitch catalogues and, where possible, against the O4a output of DANTE V3, subject to independent validation. The evaluation must avoid claiming discovery without reproducible baseline comparisons.

## 6. Minimum Viable Experiment

Use publicly available O3 or O4 data. Extract auxiliary-channel segments and compute temporal features and inter-channel coherence. Apply unsupervised anomaly-detection methods and compare recovered anomalies with known glitch families. Report parameter choices, false-positive consistency and known-family recovery rates. Record all preprocessing decisions so the experiment is reproducible.

## 7. Novelty Verification Requirements

A broader literature search beyond the previous session is required before any novelty claim. Independent replication of any reported anomaly-detection result is needed. Comparison with DANTE V3 O4a output must be conducted with full method transparency. Any candidate new behaviour must be checked against instrumental and environmental logs and described in terms of testable hypotheses rather than speculation.

## 8. Scientific Risks

Unsupervised outliers may be statistical artifacts rather than new detector behaviour. Labelling bias and catalogue incompleteness can distort comparisons. Detector state variability across O4 may affect reproducibility. Overinterpreting anomalies as physical discoveries without independent validation would violate scientific integrity. Computational costs and data-handling complexity may delay validation. Preliminary results should be communicated cautiously and clearly labelled as hypothesis-generating.
