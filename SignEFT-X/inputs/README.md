# Runtime inputs

Large or checkpoint-derived inputs are not source-controlled. The default local
run currently contains `wilor_full1493_v1/`, a target-free cache of WiLoR hand
observations for 1,493 RGB frames. It can be regenerated using the frontend
utilities documented in the project README.

Ground-truth meshes and evaluator assets must not be placed in this directory;
they are accepted only by the isolated post-hoc evaluation commands.
