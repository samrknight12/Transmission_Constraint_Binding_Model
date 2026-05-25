# Master Dataset Reference

`data/processed/master_dataset.parquet`

Built by `src/features/build_master_dataset.py` from two years of MISO Day-Ahead market data.

---

## At a glance

| Property | Value |
|---|---|
| Shape | 1,929,840 rows × 61 columns |
| Flowgates | 110 |
| Date range | 2023-01-01 00:00 EST – 2024-12-31 23:00 EST |
| Frequency | Hourly (17,544 hours × 110 flowgates) |
| Target: `binding` | 143,307 positive (7.43% overall) |
| NaN values | 0 |
| Index | `DatetimeIndex`, UTC-aware (`datetime64[ns, UTC]`) |
| Per-flowgate parquets | `data/processed/features/{flowgate_id}.parquet` |

All timestamps use MISO's fixed EST convention (UTC−5, no DST). Index stored in UTC; midnight EST = 05:00 UTC.

---

## Train / val / test split

Splits are strict calendar-day boundaries in EST, with no data shuffling.

| Split | EST range | Hours / flowgate | Rows (all fg) | Binding rate |
|---|---|---|---|---|
| Train | 2023-01-01 – 2024-06-30 | 13,128 | 1,444,080 | 7.90% |
| Val | 2024-07-01 – 2024-09-30 | 2,208 | 242,880 | 5.98% |
| Test | 2024-10-01 – 2024-12-31 | 2,208 | 242,880 | 6.03% |

Val and test rates are lower than train because summer/fall 2024 had fewer binding events than winter/spring 2023–24. No data from outside the target period leaks into any split.

Cross-validation (Optuna tuning) uses `MISOTimeSeriesSplit` with `n_splits=5`, `gap_hours=24` on the train portion only.

---

## Target variable

```
binding = 1  if  |shadow_price| > 0.01 $/MWh
         0   otherwise
```

Non-binding hours (shadow price = 0) are absent from the DA binding constraint HIST files. The pipeline reindexes each flowgate to the full hourly load-forecast index and fills missing hours with `shadow_price = 0`, producing the complete binary series.

Where a flowgate appears under multiple contingencies in the same DA hour, the maximum absolute shadow price across contingencies is used.

---

## Feature groups

60 features total (excludes `flowgate_id` identifier and `binding` target).

### Temporal (14)

Derived from the UTC DatetimeIndex, converted to America/Chicago for calendar logic.

| Column | Description |
|---|---|
| `hour_of_day` | 0–23 (Chicago local) |
| `hour_sin`, `hour_cos` | Cyclic encoding of hour (2π/24) |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `dow_sin`, `dow_cos` | Cyclic encoding of day-of-week |
| `month` | 1–12 |
| `month_sin`, `month_cos` | Cyclic encoding of month |
| `season` | 0 = winter (DJF) … 3 = fall (SON) |
| `is_weekend` | 1 on Saturday/Sunday |
| `is_peak_hour` | 1 on HE 07–22 Mon–Fri, non-holiday |
| `is_shoulder_hour` | 1 on HE 06 or 23 Mon–Fri, non-holiday |
| `is_nerc_holiday` | 1 on US Federal holidays (USFederalHolidayCalendar) |

### Load forecast (8)

Source: EIA-930 MISO balancing-authority demand forecast. Range: 53,933–127,807 MW.

| Column | Mean | Std | Description |
|---|---|---|---|
| `load_forecast_mw` | 75,218 | 11,323 | Total MISO DA demand forecast |
| `load_pct_of_peak` | 0.61 | 0.17 | Load / 365d rolling max |
| `load_deviation_from_avg_mw` | 5 | 9,545 | Load − 30d rolling mean |
| `load_deviation_from_7d_mean` | 1 | 8,929 | Load − 7d rolling mean |
| `load_change_1h_mw` | 0 | 2,201 | 1h backward diff |
| `load_change_4h_mw` | 3 | 7,925 | 4h backward diff |
| `load_change_24h_mw` | 8 | 4,318 | 24h backward diff |
| `load_forecast_ahead_1h_mw` | 0 | 2,201 | DA forecast: next-hour change (not leakage — from same DA file) |
| `load_forecast_ahead_4h_mw` | 3 | 7,925 | DA forecast: +4h change |

### Outage / thermal generation (11)

No CROW outage data available; all rows use the **gen_fuel_mix proxy** (`is_outage_proxy = 1`).
`outage_mw_total` and `thermal_da_cleared_mw` both hold Coal+Gas+Nuclear DA cleared MW (proxy for thermal capacity online). PTDF-weighted and forced/planned splits are zero in proxy mode.

| Column | Mean | Std | Notes |
|---|---|---|---|
| `outage_mw_total` | 60,035 | 13,214 | Thermal DA cleared MW (proxy) |
| `thermal_da_cleared_mw` | 60,035 | 13,214 | Same as above |
| `thermal_rt_actual_mw` | 57,803 | 12,517 | RT actual thermal generation |
| `thermal_deviation_30d_mw` | 32 | 10,785 | DA − 30d rolling mean (outage signal) |
| `thermal_rt_vs_da_gap_mw` | −2,232 | 2,523 | RT − DA (negative = under-delivery) |
| `outage_mw_change_24h` | −9 | 6,385 | 24h change in thermal MW |
| `thermal_outage_mw` | 60,035 | 13,214 | Alias of `outage_mw_total` |
| `outage_pct_of_capacity` | 71.3 | 14.7 | Thermal / 30d rolling max × 100 |
| `is_outage_proxy` | 1 | — | Constant; signals CROW data unavailable |
| `planned_outage_mw` | 0 | — | Unavailable in proxy mode |
| `forced_outage_mw` | 0 | — | Unavailable in proxy mode |

### Renewable generation (10)

Source: MISO Historical Generation Fuel Mix — DA Cleared UDS Generation (Wind, Solar).

| Column | Mean | Std | Description |
|---|---|---|---|
| `wind_forecast_mw` | 9,623 | 4,830 | DA cleared wind (all regions) |
| `solar_forecast_mw` | 963 | 1,330 | DA cleared solar (all regions) |
| `renewable_total_mw` | 10,586 | 4,789 | Wind + solar |
| `renewable_penetration_pct` | 14.5% | 7.0% | (Wind+solar) / load × 100 |
| `wind_ramp_1h_mw` | 1 | 561 | 1h backward diff of wind |
| `wind_ramp_4h_mw` | 3 | 1,887 | 4h backward diff of wind |
| `solar_ramp_1h_mw` | 0 | 424 | 1h backward diff of solar |
| `wind_ahead_1h_mw` | 1 | 561 | DA forecast: +1h wind change |
| `wind_ahead_4h_mw` | 3 | 1,887 | DA forecast: +4h wind change |
| `wind_variability_4h_mw` | 524 | 409 | 4h rolling std of wind (uncertainty proxy) |
| `wind_forecast_rolling_mae_7d` | 3,708 | 963 | 7d rolling std of wind (forecast error proxy) |

### Flowgate-specific (11)

Computed per flowgate. All rolling features use `shift(1)` — the current hour's label is never included. Verified: `flowgate_binding_freq_30d == 0` at the first binding event for all 110 flowgates.

| Column | Mean | Std | Notes |
|---|---|---|---|
| `flowgate_loading_pct` | 85.3 | 1.9 | RT loading % — 85.0 fill for unobserved hours |
| `flowgate_loading_pct_is_observed` | — | — | 1 = from RT data; 0 = 85.0 fill |
| `flowgate_loading_chg_1h` | 0 | 1.0 | 1h diff of loading_pct |
| `flowgate_loading_chg_4h` | 0 | 1.7 | 4h diff |
| `flowgate_loading_chg_24h` | 0 | 2.3 | 24h diff |
| `flowgate_distance_to_limit` | 14.7 | 1.9 | 100 − loading_pct, clipped ≥ 0 |
| `flowgate_pct_of_30d_max` | 0.95 | 0.07 | loading_pct / 30d rolling max |
| `flowgate_binding_freq_7d` | 0.07 | 0.16 | Fraction binding in prior 7d (shift(1)) |
| `flowgate_binding_freq_30d` | 0.07 | 0.13 | Fraction binding in prior 30d |
| `flowgate_binding_freq_90d` | 0.07 | 0.11 | Fraction binding in prior 90d |
| `flowgate_hours_since_binding` | 2,016 | 2,981 | Hours since last binding event (shift(1)); sentinel 8,760 if never bound |

---

## Flowgate quality tiers

Tiers classify how reliable the loading features are, based on how often RT binding data was available to back the `flowgate_loading_pct` values.

| Tier | Count | Criterion | Training behaviour |
|---|---|---|---|
| `high_signal` | 37 | observed_loading_pct ≥ 3% | All features used |
| `low_signal` | 44 | 0 < observed_loading_pct < 3% | All features kept; `loading_pct_is_observed` flags synthetic hours |
| `synthetic_only` | 29 | observed_loading_pct = 0% | `flowgate_loading_pct` and `flowgate_loading_pct_is_observed` dropped before training |

`observed_loading_pct` = fraction of the 17,544 hours where the flowgate appeared in the RT 5-minute binding data. The remaining hours receive `loading_pct = 85.0` (fill) and `loading_pct_is_observed = 0`.

### Per-flowgate summary

Sorted by binding rate descending. `obs_load` = observed_loading_pct.

| Flowgate | Binding rate | Imbalance | obs_load | Tier |
|---|---|---|---|---|
| CHAR_CK-WATFORD FLO PATINGATE-CHARLIE CK | 34.80% | 1.9:1 | 15.2% | high_signal |
| VVWNSP TR1 TR1 BASE | 23.26% | 3.3:1 | 0.0% | synthetic_only |
| VELVANSP TR1 TR1 BASE | 22.99% | 3.3:1 | 0.0% | synthetic_only |
| SWAN LK-WILMARTH FLO HELENAMN-SHEASLK | 21.31% | 3.7:1 | 5.9% | high_signal |
| WPIPSTNW TR1 TR1 BASE | 20.79% | 3.8:1 | 0.0% | synthetic_only |
| FORMAN TR12 FLO WAHPETON-HANKINSON | 16.30% | 5.1:1 | 8.9% | high_signal |
| FENOCH INTF | 16.09% | 5.2:1 | 14.2% | high_signal |
| MORRISOT-GRANTCO FLO HANK-WAP+WAP TR2 | 14.84% | 5.7:1 | 0.0% | synthetic_only |
| COOPER-ST JOE FLO ST JOE-FAIRPORT-COOPER | 14.15% | 6.1:1 | 3.3% | high_signal |
| OVER X345 XFMR FLO MCCREDIE-OVERTON | 13.27% | 6.5:1 | 7.6% | high_signal |
| PRES-TIBB 138 FLO ASTER-COMMODORE | 13.24% | 6.6:1 | 0.0% | synthetic_only |
| PRES - TIBB 138 FLO ASTER - COMMODORE 34 | 12.44% | 7.0:1 | 8.8% | high_signal |
| OAHE-SULLYBT FLO LELAND OLDS-CHAPELLECK | 12.22% | 7.2:1 | 4.0% | high_signal |
| MIDWAY5-MRYVL_SJ FLO FAIRPRT-GNTRY-NODAW | 12.05% | 7.3:1 | 4.6% | high_signal |
| GOODLAND_B_NO_1_XFMR FLO REYN-GLND&HSWD | 11.88% | 7.4:1 | 8.0% | high_signal |
| CEDARFAL TR2 TR2 BASE | 11.80% | 7.5:1 | 0.0% | synthetic_only |
| PRES-TIBB FLO PRAIRIEST-WTVERNON | 11.42% | 7.8:1 | 3.3% | high_signal |
| EDGLYTP-OAKES BASE | 11.15% | 8.0:1 | 0.0% | synthetic_only |
| PLESNTLK-LEEDS2 FLO BALTA_GRE-RAMSEY | 10.92% | 8.2:1 | 3.8% | high_signal |
| RIV-SLEM A FLO NEW BOURBON-STFRAN | 10.88% | 8.2:1 | 1.2% | low_signal |
| TEKAMAH-RAUN FLO FORT CALHOUN-RAUN | 10.67% | 8.4:1 | 5.7% | high_signal |
| MAGICCITY-SOURIS FLO LELANDOLDS-LOGAN | 10.33% | 8.7:1 | 2.9% | low_signal |
| 5BATESVI-5TALLAIP FLO CHOCTAW-CLAY | 9.79% | 9.2:1 | 3.0% | high_signal |
| TURKEY_HL-HILG FLO PRAIRIEST-WMTVERNON | 9.61% | 9.4:1 | 7.3% | high_signal |
| RAUN-FT_CAL FLO BEAVERCRFBC-GIMES | 9.48% | 9.5:1 | 1.8% | low_signal |
| MAUR_HS-CARROLTN FLO OVERTON-SIBLEY | 9.21% | 9.9:1 | 2.1% | low_signal |
| MRYVL_SJ-BRADDCTR FLO CRESTON-MARYVILLE | 8.87% | 10.3:1 | 0.0% | synthetic_only |
| LIME_CK-BARTONSS FLO HELENA-SHEAS LAKE | 8.84% | 10.3:1 | 4.3% | high_signal |
| PTBEACH-KEWAUNEE FLO FOX RVR-N APPLETN | 8.83% | 10.3:1 | 2.4% | low_signal |
| IRVINESS-BEAC_ALT FLO OTTUMWA-MONTEZUMA | 8.81% | 10.3:1 | 5.8% | high_signal |
| PVLYGRE-BYRON2 FLO BYRON-PLEASNTVLY | 8.81% | 10.3:1 | 5.6% | high_signal |
| BIGSTON-BROWN FLO ELLENDL-TWINBROOKS | 8.70% | 10.5:1 | 3.3% | high_signal |
| SUB3456-CBLUFFS FLO RACCOON TR-ARBORHL | 8.61% | 10.6:1 | 0.2% | low_signal |
| GOODLAND B NO 1 XFMR FLO REYNLDS-GOODLND | 8.46% | 10.8:1 | 9.7% | high_signal |
| CHICAGO-PRAXAIR3 FLO WC-DUM | 8.26% | 11.1:1 | 3.1% | high_signal |
| FARGO-SHEYN FLO CTR-JAMESTOWN 345 | 8.23% | 11.2:1 | 1.9% | low_signal |
| NEAST-MIRRO FLO MANRAP-SHOTO | 8.06% | 11.4:1 | 0.0% | synthetic_only |
| KNOXIND_TR1_TR1_XF | 7.64% | 12.1:1 | 0.0% | synthetic_only |
| MAPLE-08CHRYSL FLO GREENTOWN-DELCO | 7.10% | 13.1:1 | 2.1% | low_signal |
| RUGBY-RUGBY230 FLO RUGBY-BALTAJCT+GLENBO | 7.02% | 13.2:1 | 0.0% | synthetic_only |
| ASTORIA TR1_TR11 FLO BRKINGS CNTY-ASTRIA | 7.01% | 13.3:1 | 1.9% | low_signal |
| MONROE4-NSTAR1 FLO MRCCO-MNROE-MILN | 6.80% | 13.7:1 | 2.3% | low_signal |
| BATESVL TVA-BATESVL EES FLO BATESV-LSPWR | 6.80% | 13.7:1 | 3.9% | high_signal |
| ALMA2-WABACO FLO BRIGGSRD-NROCH | 6.63% | 14.1:1 | 5.9% | high_signal |
| BRV_CRK-ADAMS_I FLO BRIGSS ROAD-N ROCHST | 6.62% | 14.1:1 | 2.0% | low_signal |
| APCT-CALF FLO MCCREDIE-MONTGOMERY | 6.61% | 14.1:1 | 1.6% | low_signal |
| EASTCAMP-WALNUT 6976 FLO ECAMPUS-WALNUT | 6.57% | 14.2:1 | 4.7% | high_signal |
| EDGLYTP_EDGLYOAKES41_1_1_LN | 6.41% | 14.6:1 | 0.0% | synthetic_only |
| RISING-BONDVIL FLO RISING-SIDNEY | 6.37% | 14.7:1 | 0.0% | synthetic_only |
| MAURINE KVA1 XFMR FLO NUNDERWD-MAURINE | 6.34% | 14.8:1 | 1.8% | low_signal |
| FARGO-SHEYNNE FLO BISON-BUFFALO-JAMESTWN | 6.31% | 14.8:1 | 1.1% | low_signal |
| RILLA-RIVTON FLO ELDORADO-MTOLIVE | 6.29% | 14.9:1 | 2.5% | low_signal |
| NEBRASKA-SUB3456 FLO SUB3740-SUB3455 | 6.26% | 15.0:1 | 1.8% | low_signal |
| GR_MND-MQOKETA FLO ROCK CREEK-SALEM | 6.01% | 15.6:1 | 0.7% | low_signal |
| TEKAMAH-SUB1216 FLO FORT CALHOUN-RAUN | 5.98% | 15.7:1 | 8.3% | high_signal |
| PRINCET2-REMINGTN D FLO MONTICELLO-MAGNT | 5.93% | 15.9:1 | 5.1% | high_signal |
| LIMECK-BARTON FLO KILLDEER-QUINN | 5.56% | 17.0:1 | 3.3% | high_signal |
| FORMAN-FORMNWA FLO WAHPETON-HANKINSON | 5.55% | 17.0:1 | 3.2% | high_signal |
| BOONVIL2 9T1 WNDNG1 FLO MDISN CNTY-NRWLK | 5.54% | 17.0:1 | 3.0% | low_signal |
| AURORA-REEDS FLO EUREKA SPRGS-BVR DAM | 5.53% | 17.1:1 | 2.2% | low_signal |
| IRVINESS-BEAC_ALT FLO BONDURANT-MNTZMA | 5.47% | 17.3:1 | 4.9% | high_signal |
| DAWSONC-LEWISWP FLO BLFLD-CHRL CRK | 5.28% | 17.9:1 | 1.2% | low_signal |
| S124-FISHRHIL FLO MUDLAKE-BENTON COUNTY | 5.26% | 18.0:1 | 1.1% | low_signal |
| ABDNJCT-ELLEN FLO TWINBRKS-BIGSTNE SOUTH | 5.15% | 18.4:1 | 3.7% | high_signal |
| EAU CLA TR9 FLO EAU CLAIR T10 | 5.11% | 18.6:1 | 2.2% | low_signal |
| LINTON2_LINTOESTBM11_1_1_LN | 5.09% | 18.6:1 | 0.0% | synthetic_only |
| HELENAMN-SCOTTCO FLO CHUB LAKE-HELENA | 4.97% | 19.1:1 | 5.6% | high_signal |
| CAYUGA-08HILSDN FLO DRESSER-SUGAR CREEK | 4.90% | 19.4:1 | 0.0% | synthetic_only |
| PEL RPD-EDGETAP FLO FERGUSFL-SILVERLK | 4.84% | 19.7:1 | 1.5% | low_signal |
| CARBIDE_CARB-91_A_LN | 4.84% | 19.7:1 | 0.0% | synthetic_only |
| CREE-CRES2 FLO CRESTON-SUMMITLK N | 4.79% | 19.9:1 | 2.0% | low_signal |
| MURPHYCR-HAYWA FLO HELENA-SHEAS LAKE | 4.74% | 20.1:1 | 4.3% | high_signal |
| WEEDMAN-MAHOMET FLO CLNTN-OREANA-GSE CRK | 4.47% | 21.3:1 | 2.9% | low_signal |
| PEACGRDN PEACBRDWD23_1 1 BASE | 4.43% | 21.6:1 | 0.0% | synthetic_only |
| WOADHILL WOADHODELL11_1 1 BASE | 4.43% | 21.6:1 | 0.0% | synthetic_only |
| TEKAMAH-SUB1226 FLO FORT CALHOUN-RAUN | 4.42% | 21.6:1 | 0.0% | synthetic_only |
| ROXANA-MITTAL_2 FLO MARKTOWN-CHICAGO AVE | 4.41% | 21.7:1 | 1.8% | low_signal |
| LIME_CK-BARTONSS FLO QUINN-BLACKHAWK | 4.34% | 22.1:1 | 3.3% | high_signal |
| WISDMJCB-WISDOMCB FLO DCKSN CTY-LKFLD | 4.30% | 22.2:1 | 3.8% | high_signal |
| MIDWAY5-MRYVL_SJ FLO NODAWAY-MARYVILLE | 4.28% | 22.4:1 | 1.3% | low_signal |
| ALMA-REGAL FLO ALMA-CHEESEMAN-REGAL | 4.27% | 22.4:1 | 0.0% | synthetic_only |
| BIGSTONE-BROWNSV FLO ELLENDALE-OAKES | 4.27% | 22.4:1 | 1.9% | low_signal |
| AURORAWA-BROOKINGS FLO WHITE-SPLIT ROCK | 4.20% | 22.8:1 | 2.0% | low_signal |
| HENSEL_TR1_TR11_XF | 3.96% | 24.2:1 | 0.0% | synthetic_only |
| VERMILIO_TT2_TT28_XF | 3.94% | 24.4:1 | 0.0% | synthetic_only |
| GLEN2_GLENBPEACGRD_1_1_LN | 3.88% | 24.8:1 | 0.0% | synthetic_only |
| SWANLK-WILMARTH FLO SHEASLK-WILMARTH | 3.84% | 25.1:1 | 0.9% | low_signal |
| BANER_BANERSANDU12_1_1_LN | 3.67% | 26.3:1 | 0.0% | synthetic_only |
| NASHUA T1_H FLO NASHUA-HAWTHORN | 3.64% | 26.5:1 | 2.2% | low_signal |
| PILSBURY PILSBMAPLE23_1 1 BASE | 3.54% | 27.3:1 | 0.0% | synthetic_only |
| 7MILEBRK-PRTEDWA BASE | 3.53% | 27.3:1 | 2.4% | low_signal |
| GRANITF-BLAIR FLO WATERTOWN-APPLEDORN | 3.51% | 27.5:1 | 2.6% | low_signal |
| BIGSTON-BROWNSV FLO BRKNGS CNTY-ASTORIA | 3.49% | 27.7:1 | 1.0% | low_signal |
| HOOT_LK-FERGSFL FLO FERGUS FALS-SILVR LK | 3.44% | 28.0:1 | 0.8% | low_signal |
| 16SE-16SOUTH FLO HANNA FRANKLN | 3.43% | 28.2:1 | 2.3% | low_signal |
| MUDLAKE-VERONA 1 FLO MUDLAKE-VERONA 2 | 3.41% | 28.3:1 | 2.9% | low_signal |
| PSSF ASTR_PSSF_1763 A BASE | 3.39% | 28.5:1 | 0.0% | synthetic_only |
| MRBN TRN1 FLO MAYWOOD-HERLEMAN | 3.37% | 28.6:1 | 1.2% | low_signal |
| RYNLD4-GDLND FLO SPRAIRIE-HNYCRK-REMINTN | 3.35% | 28.9:1 | 4.8% | high_signal |
| MAYTAG3-ARORAHTS FLO JASPER-NEWTON | 3.33% | 29.0:1 | 1.5% | low_signal |
| BASS CRK-ALBANY00 FLO QUAD CITIES-ROCKCR | 3.31% | 29.2:1 | 0.0% | synthetic_only |
| LINTON2 LINTOESTBM11_1 1 BASE | 3.26% | 29.7:1 | 0.0% | synthetic_only |
| ELLIOTW-ENDERLNW FLO HANK-WAHP+WAHP T2 | 3.24% | 29.8:1 | 3.3% | high_signal |
| POWESHIEK-REASNOR FLO BONDURANT-MONTEZUM | 3.22% | 30.1:1 | 3.4% | high_signal |
| PEACGRDN_PEACGRUGBY23_1_1_LN | 3.19% | 30.4:1 | 0.0% | synthetic_only |
| BLUFFCRK-WHIG53 2FLO UNIVERSITY-MUKWONAG | 3.16% | 30.6:1 | 1.8% | low_signal |
| BARKRC-BOGLSA FLO MCKNIGHT-FRANKLIN | 3.13% | 30.9:1 | 1.9% | low_signal |
| EDEN0000-MINERALP FLO HGHLND-SPRG GRN | 3.11% | 31.2:1 | 2.1% | low_signal |
| CRESTON-ADAMSMEC FLO CRESTON-MARYVILLE | 3.03% | 32.0:1 | 0.7% | low_signal |
| BAGLEY2-WINGER FLO WILTON-WINGER | 3.02% | 32.2:1 | 0.9% | low_signal |

---

## Class imbalance by flowgate

Overall imbalance is 12.5:1 (7.4% binding). Per-flowgate ratios span 1.9:1 to 32:1,
driven by geographic exposure and seasonal dispatch patterns.

`scale_pos_weight` is computed from the **training fold only** at the start of each
XGBoost run:

| `class_weight_strategy` | Condition | XGBoost receives |
|---|---|---|
| `adjusted` | ratio > 10:1 | actual fold n_neg / n_pos |
| `mild` | 5:1 < ratio ≤ 10:1 | actual fold n_neg / n_pos |
| `none` | ratio ≤ 5:1 | 1.0 (no reweighting) |

Flowgates in the `none` category (CHAR_CK, VVWNSP, VELVANSP, SWAN LK, WPIPSTNW)
bind frequently enough that XGBoost's default loss handles class balance without
over-correction.

---

## Data quality notes

**Loading feature provenance.** MISO does not publish historical flowgate loading
percentages. `flowgate_loading_pct` is derived from RT 5-minute binding constraint
files: hours with at least one 5-minute binding interval receive
`loading_pct = 90 + 10 × (binding_intervals / 12)`; all other hours are filled
with 85.0. For 29 `synthetic_only` flowgates, 100% of hours are filled — these
constraints bind in the DA market but are rarely or never seen in the RT 5-minute
binding file over 2023–2024. Their loading columns are dropped before training.

**Outage data.** CROW system exports are not publicly available. All rows use the
gen_fuel_mix thermal proxy (`is_outage_proxy = 1`). Proxy features
(`thermal_da_cleared_mw`, `thermal_deviation_30d_mw`, etc.) signal generation
level, not offline capacity directly. PTDF-weighted and forced/planned outage
splits are zero throughout.

**Leakage guards.** All rolling binding-history features (`flowgate_binding_freq_*`,
`flowgate_hours_since_binding`) apply `shift(1)` before any window calculation.
Verified: `flowgate_binding_freq_30d == 0` at the first-ever binding event for
all 110 flowgates. No feature has |Pearson correlation| > 0.95 with the target.

**Duplicate DA hours.** Some flowgates appear under multiple contingency names in
the same DA hour. Deduplicated by taking `max(|shadow_price|)` per
(flowgate_id, datetime) before computing the binding label — if any contingency
makes the hour binding, the hour is labelled 1.

---

## Reproducing the dataset

```bash
# Rebuild from scratch (runs in ~5 min)
py -3 src/features/build_master_dataset.py

# Sanity checks
py -3 src/validation/sanity_check.py
```

Expected sanity check output: 11 PASS, 2 expected FAILs (class imbalance and
observed loading rate) reflecting known data characteristics documented above.
