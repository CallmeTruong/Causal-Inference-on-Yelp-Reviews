# Causal Inference on Yelp Reviews

**What actually drives useful votes — and what is just confounding?**

Most analyses of review helpfulness stop at correlation. This project applies **Double Machine Learning** to separate genuine causal effects from the confounding structure inherent in observational review data.

---

## Research Question

Which factors — star rating, review length, and Elite badge status — **causally** affect the number of useful votes a review receives, after controlling for user quality, sentiment, review age, and business type?

---

## Method

### Double Machine Learning (DML)

DML (Chernozhukov et al., 2018) estimates causal effects in observational data by removing the influence of confounders from both treatment and outcome before comparing the residuals. The partially linear model is:

$$Y_i = \theta \cdot T_i + g(X_i) + \varepsilon_i$$

**Estimation procedure:**
1. Fit a ML model to predict T from X → compute residual $\tilde{T}$
2. Fit a ML model to predict Y from X → compute residual $\tilde{Y}$
3. Regress $\tilde{Y}$ on $\tilde{T}$ → coefficient $\hat{\theta}$ = Average Treatment Effect (ATE)

**Implementation:** `LinearDML` from Microsoft's [EconML](https://github.com/py-why/EconML), with `HistGradientBoostingRegressor` as the outcome nuisance model and 5-fold cross-fitting.

**Causal identification assumption:** Conditional unconfoundedness — all confounders affecting both T and Y are observed and included in X.

### Handling Textual Confounding

Review text is an unstructured source of confounding: a reviewer's writing tone simultaneously shapes the star rating they give, the length they write, and how readers respond with votes. Leaving text uncontrolled conflates these signals with the treatment effects of interest.

This workflow demonstrates a general approach to this problem: **use a model to extract structured signals from unstructured text, then include those signals as explicit confounders in the causal estimator.** Here, VADER sentiment analysis converts raw review text into three numeric scores (`compound`, `pos`, `neg`) that proxy for emotional tone. These scores enter the DML confounder set X, allowing the nuisance models to partial out text-driven variation from both treatment and outcome before the causal coefficient is estimated.

The same principle extends naturally to richer representations — sentence embeddings, topic models, or LLM-based quality scores — making this a scalable pattern for causal inference on text-heavy observational data.

---

## Data

**Source:** [Yelp Open Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) — three files merged on `user_id` and `business_id`.

| Variable | Role | Description |
|----------|------|-------------|
| `stars` | Treatment | Review star rating (1–5) |
| `log_text_len` | Treatment | Log of review character count |
| `is_elite` | Treatment | Binary Elite badge status |
| `log1p(useful)` | Outcome | Log-transformed useful vote count |
| `user_review_count`, `user_useful_total`, `user_fans`, `user_friends_count` | Confounders | User quality proxies |
| `vader_compound`, `vader_pos`, `vader_neg` | Confounders | VADER sentiment scores |
| `log_review_age`, `age_bin` dummies | Confounders | Nonlinear review age encoding |
| `is_restaurant`, `is_bar`, `is_fastfood` | Confounders | Business category flags |

**Sample:** 150,000 reviews after cleaning, deduplication, and Elite reweighting.

### Key Data Issue: Elite Over-Sampling

Reservoir sampling by review (not by user) causes Elite users — who write more reviews — to appear at ~24% in the raw sample vs. their true ~5% population rate. Leaving this uncorrected inflates Elite co-occurrence in the training data and biases the ATE. The fix: downsample Elite post-merge to match the true population rate before any modeling.

---

## Results

| Treatment | ATE | 95% CI | p-value | Interpretation |
|-----------|-----|--------|---------|----------------|
| Stars (per +1) | −2.6% | [−2.85%, −2.37%] | < 0.001 | Significant, small |
| Log text length | +0.6% | [−0.10%, +1.24%] | 0.09 | Not significant |
| Elite badge | +4.2% | [+2.51%, +5.68%] | < 0.001 | Significant, small–moderate |

ATE is interpreted as percentage change in useful votes per unit increase in treatment, holding all confounders fixed: $(e^{\hat{\theta}} - 1) \times 100\%$.

### Key Findings

**Stars (negative):** After controlling for user quality, sentiment, and review age, higher star ratings causally *reduce* useful votes. This is consistent with the *information value of negative reviews* hypothesis — critical reviews contain specific, actionable details that readers find more helpful than generic praise.

**Length (null):** The raw correlation between review length and useful votes disappears entirely after controlling for user experience. Skilled users write longer reviews *and* receive more votes, but length itself has no causal effect. The "write more = be more helpful" assumption is a confounding illusion.

**Elite badge (positive):** After isolating the pure badge effect from underlying user talent, Elite status adds ~4% to useful vote count. A placebo test (Elite → funny votes: ATE = −0.1%, p = 0.91) confirms the effect is specific to credibility, not general engagement. Full-sample and propensity-score-trimmed estimates agree (+4.2% vs. +3.9%), ruling out extrapolation as a driver.

---

## Robustness Checks

| Check | Result |
|-------|--------|
| 5-seed stability (ATE std) | Stars: 0.013% · Length: 0.107% · Elite: 0.116% |
| Propensity score trimming | Elite ATE: 4.2% → 3.9% (consistent) |
| Placebo outcome (funny votes) | Elite ATE: −0.1%, p = 0.91 (no effect) |
| E-value (CI bound) | Stars: 1.18 · Length: 1.03 · Elite: 1.19 |

**E-value interpretation:** A value of ~1.2 means an unobserved confounder would need to have a risk ratio of at least 1.2 on both T and Y to fully explain away the result. All estimates are statistically valid but sensitive — results should be interpreted as directional evidence, not definitive causal claims.

---

## Stack

| Component | Tool |
|-----------|------|
| Causal framework | EconML `LinearDML` |
| Outcome nuisance model | `HistGradientBoostingRegressor` (sklearn) |
| Treatment model (binary) | `LogisticRegressionCV` (sklearn) |
| Treatment model (continuous) | `RidgeCV` (sklearn) |
| Sentiment extraction | VADER (`vaderSentiment`) |
| Propensity score | `LogisticRegression` (sklearn) |
| Sensitivity analysis | E-value (VanderWeele & Ding, 2017) |
| Data | Yelp Open Dataset (Kaggle) |

---

## Limitations

- **Low E-values (~1.1–1.3):** Results are sensitive to unobserved confounders. Yelp's internal ranking algorithm, image quality, and niche expertise are plausible confounders that could shift conclusions.
- **No instrument for Elite status:** Without a valid instrumental variable, the unconfoundedness assumption cannot be fully tested.
- **Static analysis:** The dataset spans multiple years; user behavior and voting norms may have shifted over time.
- **Platform-specific:** Findings are calibrated to Yelp's user culture and badge system and may not generalize to other review platforms.

---

## Future Directions

1. **Text embeddings or LLM-scored content quality as confounders** — the single most impactful missing control; would likely increase E-values substantially.
2. **`CausalForestDML` for heterogeneous treatment effects** — does the Elite badge matter more for newcomers? For restaurants vs. services?
3. **Instrumental variable for Elite status** — would remove reliance on the unconfoundedness assumption and strengthen causal identification.

---

## References

- Chernozhukov, V. et al. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1).
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity Analysis in Observational Research: Introducing the E-Value. *Annals of Internal Medicine*, 167(4).
- Hutto, C., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. *ICWSM*.
