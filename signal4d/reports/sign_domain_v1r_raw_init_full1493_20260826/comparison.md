# Author SGNify evaluator comparison

Protocol: `manifest`. Values are the author's vertex-micro means in mm; lower is better.

| Method | TR all | TR upper body | TR left hand | TR right hand |
|---|---:|---:|---:|---:|
| DexAvatar | 42.5867 | 26.4560 | 13.5735 | 12.9271 |
| ExternalV1 | 42.2423 | 26.2236 | 12.8102 | 12.1148 |
| SignDomainClean | 42.2433 | 26.2249 | 12.8102 | 12.1141 |
| SignDomainRawInit | 37.9349 | 25.9246 | 21.2501 | 20.4510 |

Deltas in `comparison.csv` are method minus `DexAvatar`; negative is better.
