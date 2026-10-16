# Parkwise: product brief

## The problem
Parking operators price with a static schedule, so busy lots turn drivers away while quiet lots sit
empty. Drivers who find a full lot after trusting the app are the most expensive failure: they churn.

## Who it is for
- **Pricing and revenue managers** who set zone prices and need to defend them.
- **Operations teams** who need to know which lots will fill and when.
- **Support leads** who see the customer cost of a price change first.

## What it does
1. Forecasts demand for every lot, hour by hour, one day ahead, with an 80% range.
2. Learns how sensitive drivers in each lot are to price, from randomised price tests.
3. Recommends hourly prices under guardrails the business controls.
4. Projects the customer cost of a plan (complaints) next to the revenue gain.

## Success metrics
| | Metric | Why |
|---|---|---|
| North star | Net value per bay-day: revenue minus the goodwill cost of drivers turned away or priced out | Revenue alone rewards pricing drivers out |
| Guardrails | Overcharge complaints per 10k parked hours; drivers priced out; average price vs schedule; hour-to-hour price change | Each protects a different kind of trust |
| Model health | WMAPE vs last week's pattern; 80% range coverage; full-lot warning precision and recall | Detects drift before it costs money |

## Rollout plan
1. **Shadow (2 weeks).** Run forecasts and recommended prices without charging them. Compare forecast error
   with the offline numbers.
2. **Pilot (4 weeks).** 3 to 4 zones, randomised by day with a holdout. Primary readout: net value per bay-day.
   Stop rules: complaints above 2x baseline, or priced-out drivers above 25% over holdout.
3. **Expand** zone by zone as each passes. Keep a standing 10% holdout and the randomised price tests, which are
   what keep the sensitivity estimates honest.

## Risks and what the project already shows about them
| Risk | Evidence / mitigation |
|---|---|
| Revenue gains come from pricing drivers out | Priced-out drivers rise 75% in the replay. The goodwill terms and caps are controls, and the trade-off is on every screen |
| Price complaints | +25% price gives about 7.6x overcharge complaints in the data. Hourly change limit and daily average cap exist for this |
| Elasticity is mis-estimated | Overestimating sensitivity by 50% still keeps +9.1% net; underestimating is nearly free in net terms (see Model health) |
| Demand shifts (new competitor, new tower) | The simulator includes two level shifts. History features carry the adjustment (removing them costs 1.7 pts of error); range coverage below 80% is the signal to monitor |
| New lots have no history | Area-mix forecasts halve the error where the mix exists elsewhere; unique archetypes need fast adaptation |
| Full lots hide true demand | Log drivers turned away. Occupancy-only estimates understate sensitivity (Koramangala -0.72 vs -1.00) |

## Roadmap
- Replace the static spill assumption with learned substitution between lots.
- Per-segment price sensitivity (commuters vs visitors) and time-varying elasticity.
- Complaint cost inside the optimiser objective, not only as a projection.
- Rolling recalibration of the forecast range; drift alerts.
