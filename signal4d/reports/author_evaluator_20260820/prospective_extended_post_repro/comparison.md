# Author SGNify evaluator comparison

Protocol: `manifest`. Values are the author's vertex-micro means in mm; lower is better.

| Method | TR all | TR upper body | TR left hand | TR right hand |
|---|---:|---:|---:|---:|
| A1 | 43.0174 | 29.2938 | 14.4918 | 12.8413 |
| M1_alpha1 | 42.9994 | 29.2720 | 14.0304 | 12.8420 |
| M1_alpha1p5 | 43.0613 | 29.3597 | 14.2332 | 12.8430 |
| M1_alpha3 | 43.5322 | 29.9908 | 16.4099 | 12.8463 |
| SIGNAL4D_v5 | 43.0072 | 29.2774 | 14.0354 | 12.8425 |

Deltas in `comparison.csv` are method minus `A1`; negative is better.
