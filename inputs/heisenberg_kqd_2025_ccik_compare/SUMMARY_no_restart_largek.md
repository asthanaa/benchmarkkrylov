# No-Restart Large-K Follow-up

No-restart large-K CCIK does not beat experimental KQD at same D.

## Setup

- Single-cycle local solver only: `m_cycle=10`, no thick restart.
- Requested `Kstore` grid: `[100000, 150000, 300000, 500000, 700000, 'full_dim']`.
- `Kprop/Kstore` ratios: `[0.5, 1.0]`.
- Best envelope is chosen independently at each D over the scanned `(Kprop, Kstore)` settings.

## Final-D10 Verdict

- `N56_k1`: best `D=10` abs error `0.231454154694` with `(Kprop,Kstore)=(28,56)`; paper KQD `D=10` abs error `2.23881113329`; beats=`True`
- `N44_k3`: best `D=10` abs error `0.309966915527` with `(Kprop,Kstore)=(6622,13244)`; paper KQD `D=10` abs error `3.23763618775`; beats=`True`
- `N42_k5`: best `D=10` abs error `3.42386763789` with `(Kprop,Kstore)=(100000,100000)`; paper KQD `D=10` abs error `2.38387900566`; beats=`False`

## Bottom Line

The blocker remains `N42_k5`: even the best scanned no-restart large-K setting ends at abs error `3.42386763789`, while the paper KQD `D=10` error is `2.38387900566`.
So large `Kprop/Kstore` alone is not enough to make the no-restart solver beat the paper in all three cases.
