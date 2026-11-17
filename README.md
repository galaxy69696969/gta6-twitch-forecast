[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/galaxy69696969/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/galaxy69696969/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch forecast

Entry for make.fun's [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty.

`main.py` (Python 3.13, standard library only, no network, deterministic)
forecasts Twitch's concurrent viewers in the Grand Theft Auto VI category
for each hour 2–24 of release day, 2026-11-19 (UTC), as five quantiles
per hour.

## How it works

For every comparable past launch shipped in the case file, the model
anchors on that launch's hour-1 print and works in log ratios
`ln(viewers_h / viewers_1)`. The target's forecast for each hour comes
from a Gaussian fit (weighted mean and spread) of those ratios, with
each comparable weighted by a Gaussian kernel on `ln(hour-1 viewers)` —
so launches the size of the target dominate — and early-access launches
(high pre-release baseline) down-weighted when the target itself looks
like a clean launch. The log-spread is floored at 0.30 so the bands
never claim more precision than a few dozen launches earn. See the
docstring at the top of `main.py` for the full description and the
local cross-validation numbers.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `<id>.out`
per case.
