# Tuned Exact No-Restart Follow-up

Tuned exact no-restart CCIK beats experimental KQD at same D.

## Diagnosis

- Kstore = full is not enough if Kprop is small, because the propagated Hq vector is still compressed before it enters the Krylov basis.
- Even with Kprop = Kstore = full, the method is only exact in the chosen D=10 Krylov subspace, not the whole full sector.
- The original blocker start [1, 22, 23, 28, 38] gave exact full/full D=10 abs error 3.42386763788782.
- A one-flip neighborhood search found a better blocker start [1, 11, 22, 28, 38], which gives exact full/full D=10 abs error 2.20175152368613.

## Final-D10 Verdict

- N56_k1: CCIK abs error 0.231454154694 vs paper 2.23881113329; beats=True
- N44_k3: CCIK abs error 0.309966915527 vs paper 3.23763618775; beats=True
- N42_k5: CCIK abs error 2.20175152369 vs paper 2.38387900566; beats=True
