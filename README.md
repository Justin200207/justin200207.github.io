# Developer Salary Explorer — MIE1624 Assignment 2

A complete static dashboard for GitHub Pages based on the **actual** final ordinal logistic regression model from the assignment. The site asks 17 **original** Stack Overflow survey questions, predicts a probability for every one of the 15 ordered USD salary ranges, and ranks skills/tools by the change in expected ordinal bucket when the tool is added to the profile.

## Deploy to GitHub Pages

1. Create a public GitHub repository (e.g. `developer-salary-explorer`).
2. Upload the **contents of this folder** to the repository root. `index.html` and `model.json` must both be in the root.
3. Commit and push the files to the `main` branch.
4. On GitHub, open **Settings → Pages → Build and deployment**. Set **Source: Deploy from a branch**, **Branch: main**, and **Folder: /(root)**. Save.
5. Open the URL GitHub provides, normally `https://YOUR-USERNAME.github.io/developer-salary-explorer/`.

**Important:** Upload `index.html` with the correct spelling, *not* `index.htlm`. The GitHub Pages root automatically serves `index.html`. Do **not** upload the zip file itself as the only repository content; extract it first.

## Preview locally

The dashboard uses an ordinary JavaScript module and fetches `model.json`. Modern browsers block JSON fetches from a `file://` URL. Preview it over an HTTP server:

```bash
cd salary-dashboard
python -m http.server 8000
```

Open `http://localhost:8000/`.

## Model provenance

- Original survey input: `processed_data_assignment2_2026.csv` (not included in the deployable website).
- The final sklearn pipeline uses the 18 selected features from Q2, of which **17** are original questions and `WorkExpSquared` is derived automatically from `WorkExp`.
- Categorical answers use training-fitted `OneHotEncoder(handle_unknown="ignore")` and the notebook's custom semicolon-delimited `MultiSelectEncoder`. Numerical features are median-imputed and standardized; missingness flags are preserved.
- Model: **14 cumulative-threshold LogisticRegression classifiers**, `solver="liblinear"`, `penalty="l1"`, `C=0.3`, `random_state=42`, `max_iter=500`.
- Browser model is an export of the real intercepts, coefficients, training-fitted vocabularies and numerical preprocessing statistics. No prediction API or backend is needed. Nothing is trained or fitted on visitor responses.
- The cumulative probabilities are clipped and made nondecreasing using the cumulative maximum. Differences produce 15 normalized class probabilities. The predicted bucket is the distribution's **median** (for minimizing absolute ordinal error).
- Recommender: evaluate each unselected option in `LanguagesHaveWorkedWith`, `CloudPlatHaveWorkedWith` and `KnowMgmtHaveWorkedWith` with **training support ≥ 30**. Keep all other responses unchanged. Rank by **change in the expectation of ordinal bucket codes 0–14**, and display positive changes. Recommendations are associations, **not causal effects**.
- Reported hold-out metrics reproduced during export: training MAE = 1.2121; test MAE = 1.3179; test accuracy = 32.53%; within-one-bucket = 69.99%.

> Limitations: This is observational survey data, and category sizes are unequal in USD. Neither bucket index changes nor model probabilities are guaranteed salary gains, currency-valued forecasts, or calibrated prediction intervals. The model is weakest on rare salary classes.

## Files

- `index.html` — page markup, including all dashboard sections.
- `styles.css` — responsive styling.
- `app.js` — original-survey form, chart rendering and interactivity.
- `model-core.js` — exact numerical predictor and recommendation logic.
- `model.json` — fitted sklearn coefficients and preprocessing statistics (not the training data).
- `scripts/export_model.py` — optional reproducible exporter; run again with `--csv /path/to/processed_data_assignment2_2026.csv` if you need to regenerate the model.

Run the lightweight browser-core tests with `npm test` (Node 20+). No `npm install` required.

## Privacy

Survey answers stay in your browser's current page session and are never transmitted to a model server. This static site does not use cookies or analytics. Opening the site downloads a shared, fitted model file; **no individual survey responses are included in the deployment**.

## Academic attribution

MIE1624 — Introduction to Data Science, Analytics and AI, Fall 2026. Model and findings from Assignment 2. Underlying data: 2026 Stack Overflow Developer Survey.
