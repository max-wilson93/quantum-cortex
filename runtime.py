"""Pin BLAS to one thread. Import this before numpy, from every entry point.

This model's inference is a *stream of small matrix-vector products*: one
3136x50 complex matvec per sample per column, plus four 50x50 ones. At that
size, handing the work to a multithreaded BLAS costs more in thread
synchronisation than it saves in arithmetic. Measured on a 4-core machine:

    3136x50 complex matvec    4 threads: 414 us     1 thread:  91 us
    one training sample       4 threads: 5.21 ms    1 thread: 0.73 ms

So the model runs about 7x faster on one thread than on four. That is not a
tuning detail, it is a property of the architecture worth stating: the
computation is latency-bound on small operations rather than throughput-bound
on large ones. That is exactly the regime a per-sample online learner lives in,
and exactly why roadmap 4.4 wants the efficiency claims measured rather than
asserted -- a "low compute cost" argument that ignores where the time actually
goes is not an argument.

It also means the honest way to use several cores here is to run several
independent configurations at once, not to split one matvec across them.

Values are set with setdefault, so an explicit environment variable still wins.
"""

import os

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

for _var in _THREAD_VARS:
    os.environ.setdefault(_var, "1")
