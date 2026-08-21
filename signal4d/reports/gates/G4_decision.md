# G4 contact-data decision

No independent frame/event contact annotations, inter-rater agreement, or label-uncertainty audit are available for the 57-clip SGNify endpoint. Therefore G4 fails for a real-data contact-correctness claim.

The deterministic synthetic validation (three clips, 24 frames each) gives:

- frame contact F1: 1.000, 0.909, 0.667 (mean 0.859);
- event macro F1: 1.000, 1.000, 0.833 (mean 0.944);
- one worst-case sticky offset error of four frames.

Decision: retain M2 as an exploratory proximity/collision and failure-diagnostic module. Report active fraction, switch rate, and collision-proxy penetration separately. Do not use these proxies as contact ground truth and do not claim improved contact reconstruction.
