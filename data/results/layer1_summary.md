## Layer 1 Results -- MISO Transmission Constraint Classification

*Generated: 2026-05-25*

---

### Model Coverage

- Flowgates evaluated (production + marginal): **72**
- Production models (val PR-AUC >= 0.70): **51**
- Marginal models (val PR-AUC 0.40-0.69): **21**
- Regime-change / low-signal (excluded from evaluation): 31
- Dropped (insufficient signal): 6

| Metric                            | Value   |
|-----------------------------------|---------|
| Median test PR-AUC (production)   | 0.8027  |
| Median test PR-AUC (marginal)     | 0.5514  |
| Top-20 hourly precision           | 24.9%    |
| Median Brier score                | 0.0384  |
| Median ECE (production, raw)      | 0.0863  |
| Models requiring calibration      | 32 / 51  |

---

### Tier Breakdown

| Tier           | Count | Median test PR-AUC | Median Brier |
| -------------- | ----- | ------------------ | ------------ |
| high_signal    | 21    | 0.8339             | 0.0438       |
| low_signal     | 24    | 0.7905             | 0.0430       |
| synthetic_only | 18    | 0.6210             | 0.0666       |

*Rows with 0 test-window binding excluded from PR-AUC / Brier medians (9 models).*

*synthetic_only tier ceiling is lower by design: these models have no observed flowgate loading data and rely entirely on system-level features (load, renewables, outages, calendar). The 0.621 median reflects the information limit of public data for these constraints, not a modelling deficiency.*

---

### Notable Models

Top 5 by test PR-AUC (flowgates with at least 1 binding hour in test window):

| Flowgate                                | Status     | Test PR-AUC | Test F1 | Binding hrs / Total |
| --------------------------------------- | ---------- | ----------- | ------- | ------------------- |
| OAHE-SULLYBT FLO LELAND OLDS-CHAPELLECK | Production | 0.9708      | 0.9391  | 307 / 2208          |
| APCT-CALF FLO MCCREDIE-MONTGOMERY       | Production | 0.9591      | 0.9333  | 144 / 2208          |
| ALMA-REGAL FLO ALMA-CHEESEMAN-REGAL     | Production | 0.9497      | 0.9406  | 198 / 2208          |
| 7MILEBRK-PRTEDWA BASE                   | Production | 0.9390      | 0.8889  | 37 / 2208           |
| RAUN-FT_CAL FLO BEAVERCRFBC-GIMES       | Production | 0.9326      | 0.9180  | 259 / 2208          |

---

### Regime Change Findings

These flowgates have high cross-validation PR-AUC on training folds but zero binding events in the Jul-Sep 2024 validation window. Monthly analysis confirmed a structural regime change in Q1 2024 (network reconfiguration or OLR revision), not seasonal variation. Val PR-AUC is permanently 0 for this window; CV PR-AUC is the quality signal.

| Flowgate                                 | CV PR-AUC | Test binding hrs | Notes                                                |
| ---------------------------------------- | --------- | ---------------- | ---------------------------------------------------- |
| CHAR_CK-WATFORD FLO PATINGATE-CHARLIE CK | 0.7728    | N/A              | Bakken load pattern, partial return Oct 2024         |
| MORRISOT-GRANTCO FLO HANK-WAP+WAP TR2    | 0.8918    | N/A              | Binding collapse Q1 2024, likely topology/OLR change |
| TURKEY_HL-HILG FLO PRAIRIEST-WMTVERNON   | 0.7999    | N/A              | Binding collapse Q1 2024, likely topology/OLR change |
| GR_MND-MQOKETA FLO ROCK CREEK-SALEM      | 0.8998    | N/A              | Binding collapse Feb 2024, zero from Mar 2024 onward |

---

### Data Coverage

| Period     | Dates                          | Role                     |
|------------|--------------------------------|--------------------------|
| Training   | 2023-01-01 to 2024-06-30       | Optuna CV + final fit    |
| Validation | 2024-07-01 to 2024-09-30       | Threshold + calibration  |
| Test       | 2024-10-01 to 2024-12-31       | Held-out, never fit      |

**Features**: 59 total across 5 groups

  - **Generation & renewables**: 16 features
  - **Time & calendar**: 15 features
  - **Flowgate loading & history**: 10 features
  - **System load**: 9 features
  - **Outages (MW & count)**: 9 features

All timestamps UTC. MISO EST = UTC-5 (fixed, no DST adjustment).

**Dropped from scope**: 6 flowgates marked `insufficient_signal` (CV PR-AUC < 0.20 with zero val binding; excluded from all downstream layers).