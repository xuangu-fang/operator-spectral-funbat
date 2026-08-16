# Aborted development audit

These four files are intentionally retained as a negative engineering audit,
not as a completed experiment sweep.  They used collapsed coefficients, a 10%
generic floor, uniform eight-way initialization, and a temporary routing-logit
learning rate of 0.1 on already-exposed development seeds 101--104.

The sweep was stopped before seed 105 because matched robust routing was highly
seed-unstable (`0.033, 0.724, 0.047, ...` NRMSE). Its apparent strict-support
failure was later traced to the atom-derived zero-basis bug and is invalid. No
statistic from this directory enters the submission table.
The single minimal retry with the original common learning rate, a 25% floor,
and operator-centred initialization is in
`../escape_floor_operator_init_development/`.
