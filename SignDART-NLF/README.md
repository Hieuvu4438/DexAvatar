# SignDART-NLF

Isolated implementation of the gated SignDART-NLF research lane. The code
reads immutable H1 artifacts and never mutates DexAvatar, SignEFT-X, or the
official evaluator.

The implementation follows the gates in
`docs/proposal12/SignRay_X_Deep_Research_Implementation_v4 (1).md`:

1. reproduce and hash-lock the H1 incumbent;
2. validate finite ray--sphere arm candidates and invariants;
3. measure the development-only candidate oracle ceiling;
4. run NLF extraction/selection only if the oracle gate passes.

