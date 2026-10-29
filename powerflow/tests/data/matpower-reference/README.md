# MATPOWER DC power flow reference results

Expected results for the `powerflow` DC power flow tests: bus voltage angles, branch flows and
generator outputs computed by MATPOWER's `rundcpf`. The versions used are in
[`VERSIONS`](VERSIONS).

These files are **generated**. Don't edit them by hand; regenerate them with
`scripts/generate-reference.sh`, which fetches MATPOWER (`scripts/fetch-matpower.sh`) and runs
`scripts/generate_reference.m` under Octave. The output is deterministic, so a regeneration
with the same versions leaves no diff.

## Files

For each case:

| File | Columns |
|---|---|
| `<case>_dcpf_bus.csv` | `bus` (bus number), `va_deg` (voltage angle, degrees) |
| `<case>_dcpf_branch.csv` | `row` (1-based row in the case file's branch matrix), `from`, `to`, `status`, `pf_mw` (real power flow at the from end, MW) |
| `<case>_dcpf_gen.csv` | `row` (1-based row in the gen matrix), `bus`, `pg_mw` (real power output, MW) |

Values have 17 significant digits, so they parse back to the exact doubles MATPOWER computed.
In a DC power flow the flow at the to end is `-pf_mw`.

## Sources and attribution

The results were computed with MATPOWER:

> R. D. Zimmerman, C. E. Murillo-Sánchez and R. J. Thomas, "MATPOWER: Steady-state operations,
> planning, and analysis tools for power systems research and education," *IEEE Trans. Power
> Syst.* 26(1):12–19, 2011. [doi:10.1109/TPWRS.2010.2051168](https://doi.org/10.1109/TPWRS.2010.2051168)

MATPOWER's code is under the BSD 3-clause license. Its case files are not covered by that
license, so they are not included in this repository; the fetch script downloads them. Only
results are committed here, and only for cases that come from public sources, as recorded in
each case file's header:

| Case | Origin |
|---|---|
| `case14` | IEEE 14-bus test case (1962), converted by MATPOWER from the IEEE Common Data Format file in the University of Washington Power Systems Test Case Archive, <https://labs.ece.uw.edu/pstca/>. |
| `case30` | Compiled by MATPOWER's authors from O. Alsaç and B. Stott, "Optimal load flow with steady-state security," *IEEE Trans. Power App. Syst.* PAS-93(3):745–751, 1974, [doi:10.1109/TPAS.1974.293972](https://doi.org/10.1109/TPAS.1974.293972), and R. W. Ferrero, S. M. Shahidehpour and V. C. Ramesh, "Transaction analysis in deregulated power systems using game theory," *IEEE Trans. Power Syst.* 12(3):1340–1347, 1997, [doi:10.1109/59.630479](https://doi.org/10.1109/59.630479), with modifications described in the case file. |
| `case118` | IEEE 118-bus test case (1961), from the same University of Washington archive, with base kV data and two transformer corrections added by MATPOWER's authors. |

The larger `case2383wp` (Polish grid), used for benchmarks, is distributed by MATPOWER with
its data provider's permission. No results derived from it are committed; anything that needs
it fetches MATPOWER.
