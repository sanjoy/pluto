# External research dependencies

## lp_solve

The one-shot memorization experiments use the C API of
[lp_solve 5.5.2.14](https://github.com/lp-solve/lp_solve/releases/tag/5.5.2.14)
for continuous linear programs. Its revised-simplex method pivots between linear
program bases; it is not gradient descent on the neural-network loss. It remains
an iterative numerical algorithm, not a closed-form formula or a guarantee of
exact arithmetic.

`MODULE.bazel` fetches the unchanged, SHA-256-pinned upstream source archive. The
release tag identifies commit `305e32cbfae961d580336258caeaeaa7c5538b91`.
`lp_solve.BUILD.bazel` follows the source list and definitions in upstream
`lpsolve55/ccc`. The public Bazel target is `@lp_solve//:lp_solve`, and its API
header is `lp_lib.h`. Sources compile as C without C++ exceptions. The archive
supplies LUSOL, COLAMD, and BLAS helper routines; this Linux configuration only
needs the system math and dynamic-loader libraries.

The solver is distributed under the
[GNU Lesser General Public License 2.1](https://github.com/lp-solve/lp_solve/blob/5.5.2.14/LICENSE).
The archive retains upstream notices, including notices for bundled components
and the generated parser's license exception. Distribution of linked binaries
must meet the applicable license requirements; fetching the library externally
does not remove those obligations.

Bundled COLAMD attribution: Stefan I. Larimore and Timothy A. Davis,
University of Florida; Copyright (c) 1998–2001 by the University of Florida.
Used by permission. Its original notices and availability information are
preserved in the unmodified upstream `colamd/colamd.c` and `colamd/colamd.h`.
