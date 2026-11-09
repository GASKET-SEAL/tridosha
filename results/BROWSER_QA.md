# Browser verification

Date: 9 November 2026. Local Python server, in-app Chromium browser.

| Check | Observed result |
| --- | --- |
| Default load | Python result displayed: stable; 2.6-day controlled recovery; 6.4-day natural recovery; 49.6% integrated-error reduction |
| Equilibrium scenario | Zero deviation; 34%, 33%, 33% shares; zero recovery time; no invalid percentage division |
| Pitta scenario | 2.8-day controlled and 7.4-day natural recovery; 51.9% reduction |
| Time scrubber | Day 10 snapshot and triangle marker updated; distance approximately 0.006 |
| Invalid pulse | A 30-day duration within a 21-day horizon produced a validation message |
| Reset | After renaming a button that masked HTMLFormElement.reset, reset restored duration from 30 to 2 and reran default results |
| Export | Downloaded tridosha-experiment.json containing completed configuration and both trajectories |
| Desktop | Inspected at 1440 x 1050; final screenshot included in report folder |
| Mobile | Inspected at 390 x 844; document width 375 within 390-pixel viewport, with no horizontal overflow |
| JavaScript | node --check web/app.js completed successfully |

This is an observed interaction checklist, not an automated cross-browser certification. The reset failure was fixed before final delivery. No accessibility conformance or production load-test claim is made.
