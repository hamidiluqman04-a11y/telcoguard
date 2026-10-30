# TelcoGuard user guide

This guide explains **who TelcoGuard is for**, **what it can and cannot tell
you**, and **everything you need to know to use its results responsibly**.

For installation, commands and technical design, see the
[README](../README.md).

---

## 1. What TelcoGuard is, in one paragraph

TelcoGuard estimates how likely each telecom customer is to **cancel their
service ("churn")**. It is based on account details such as contract type,
tenure, services and billing. It groups customers into risk levels, shows
which factors drive each estimate, and lists retention ideas worth
**testing**. It was trained on a historical sample of 7,043 customers, and it
runs on your own computer as a set of interactive marimo notebooks.

---

## 2. Who should use it

| You are… | You will use it to… | Start with | Skills needed |
|---|---|---|---|
| **Retention / customer-success manager** | find at-risk customers, decide who to contact, and choose which offers to trial | Notebook 05 (insights app) | Reading tables and charts. No coding. |
| **Marketing or CRM analyst** | export target lists, size campaigns, and compare scenarios with your own costs | Notebooks 05 and 04 (thresholds and costs) | Spreadsheets/CSV; basic statistics helps |
| **Business leader / decision-maker** | understand how reliable the predictions are and what they cost or save | Section 1 of Notebook 05, Section 7 of Notebook 04, and this guide | None |
| **Data scientist / ML engineer** | audit, retrain, extend or deploy the model | All notebooks, `src/`, `tests/` | Python, scikit-learn |
| **Student or learner** | learn a complete, leakage-safe ML workflow end to end | Notebooks 01 → 05 in edit mode | Basic Python |

### Who should *not* rely on it (as-is)

- **Teams wanting fully automated decisions.** TelcoGuard is a
  decision-*support* tool. A person should review who is contacted and what
  they are offered.
- **Anyone making decisions that could harm customers.** Examples are
  refusing service, raising prices or changing credit terms based on churn
  risk. The model was not built or reviewed for that.
- **Companies whose customers look different from the training data.**
  Different products, markets, price levels or time periods may give
  unreliable scores. Validate on your own labelled data first (section 9).
- **Anyone who needs to know *why customers leave* in a causal sense.** The
  model finds patterns, not causes (section 4).

---

## 3. What you need before you start

- A computer with **Python 3.13** and the project installed (see the README).
  A one-time setup takes a few minutes.
- The project's notebooks **02 → 04 run at least once**, because they create the
  model files. On a new copy of the project this takes about 2 minutes. If you
  skip it, the app tells you exactly which notebook to run.
- For your own customers: a **CSV file in the same format** as the original
  dataset (section 9).

**Quick start for non-technical users.** Ask whoever installed the project to
run this command, then open the `http://localhost:…` link it prints in your
browser (it usually opens automatically):

```bash
.venv/Scripts/marimo.exe run notebooks/05_churn_insights.py
```

You will see an app with no code. Everything is controlled with dropdowns,
sliders, a table and buttons.

---

## 4. What it can and cannot tell you

| ✅ It **can** tell you | ❌ It **cannot** tell you |
|---|---|
| Which customers *look like* past customers who churned | Whether a specific customer *will* churn |
| Roughly how many churners to expect in a group | The exact names of who will leave |
| Which account factors the model relies on most | What *causes* customers to leave |
| Which offers are *associated* with lower churn, and how the model's estimate changes | Whether an offer will actually *retain* a customer |
| How accurate it was on 1,409 held-out customers | How accurate it will be next year, or for another company |

### The two most important ideas

**1. A risk score is an estimate, not a certainty.** "70% churn risk" means
that among many *similar* customers, about 7 in 10 left in the past. The
customer in front of you may well stay. A "10% risk" customer may leave. Use
scores to **prioritise**, not to predict individuals.

**2. Association is not causation.** Customers on two-year contracts churned
far less (about 3%) than month-to-month customers (about 43%). That does
**not** mean switching someone to a two-year contract will stop them leaving.
People who *choose* long contracts are different in many other ways: they
may be more settled or more satisfied. The only way to learn whether an offer
works is to **test it** (section 7).

---

## 5. How good is the model? (in plain numbers)

Measured once on **1,409 customers the model never saw during training**:

- It **catches about 3 out of 4 churners** (recall 77.5%).
- About **half of the customers it flags** (53.6%) actually churned. The
  other half would have stayed.
- It flags **38%** of customers in total.
- Out of 374 real churners, **290 were caught and 84 were missed**. 251
  customers were flagged unnecessarily.

The risk groups behave as intended on these test customers:

| Risk group (default cut-offs) | Customers | Actually churned |
|---|---|---|
| Low (below 30%) | 868 | 9.7% |
| Medium (30–60%) | 354 | 43.5% |
| High (60% and above) | 187 | 72.7% |

Across all 1,409 test customers, the model expected **374** churners, and
**374** actually churned. The probabilities are well calibrated *on
average*.

> These numbers come from one historical dataset. Expect them to be somewhat
> different, and usually a little worse, on new data.

---

## 6. Reading the results: a plain-language glossary

| Term | What it means | How to use it |
|---|---|---|
| **Churn probability / risk** | The model's estimate (0–100%) that a customer will cancel | Sort customers and focus on the highest first |
| **Threshold (0.30)** | The cut-off above which a customer is **flagged** | Lower it to catch more churners (with more false alarms); raise it for fewer, more certain flags |
| **Flagged** | Risk at or above the threshold | Candidates for retention outreach |
| **Risk group (Low / Medium / High)** | Bands of risk with adjustable cut-offs | Match the size of each band to your team's capacity |
| **True positive** | Flagged, and did churn | A churner you could try to keep |
| **False positive** | Flagged, but would have stayed | Retention effort spent unnecessarily |
| **False negative** | Not flagged, but churned | A churner you missed |
| **Precision** | Of the flagged customers, the share who really churn | Higher means less wasted effort |
| **Recall** | Of all churners, the share the model flags | Higher means fewer missed churners |
| **ROC-AUC / PR-AUC** | Overall ranking quality (higher is better) | Mainly for comparing models; you don't need them for daily use |
| **Expected churners** | The sum of all probabilities in a group | The best estimate of *how many* will leave, though not *who* |
| **Feature importance** | How much the model relies on each column overall | Tenure and contract type matter most |
| **Customer explanation** | How a customer's risk would change if one attribute were "typical" | Shows *why the model* scored this customer high or low |
| **Model risk after (what-if)** | The model's estimate if one attribute changed, with everything else, including price, held fixed | A hint about which offer to test, **not** a promise |

---

## 7. A recommended workflow for a retention campaign

1. **Score customers.** Open the insights app and choose your customers
   (test customers, or upload your own CSV).
2. **Set capacity.** Move the risk cut-offs until the number of High (or
   Medium + High) customers matches how many your team can contact.
3. **Filter and export.** Narrow the list, for example to month-to-month
   fibre customers, then use **Download filtered** to get a CSV for your CRM.
4. **Review individuals.** Click a few customers to check the explanations
   make sense. Look out for data errors, such as unusual charges.
5. **Choose offers to test.** Use the segment view (section 8 of the app) to
   see which offers apply to the most flagged customers.
6. **Run a controlled test. This step matters most.** Randomly split the
   flagged customers into a group that gets the offer and a **control group
   that does not**. Only the difference between the two groups' churn shows
   what the offer actually achieved.
7. **Use your real economics.** In Notebook 04, section 5b, enter your real
   cost per contact, the value of a customer, and the save rate you measured
   in step 6. Adjust the threshold based on that rather than on the default.
8. **Repeat** each period with fresh data, and keep checking accuracy
   (section 11).

---

## 8. Using the insights app

| Section | What you do there |
|---|---|
| 1. The model | Read the model's track record. Note that about half of flagged customers would not have churned. |
| 2. Choose customers | Pick **Test customers** for honest scores, **All customers** (scores are optimistic for the 80% used in training), or **Uploaded CSV** |
| 3. Risk overview | Adjust the cut-off slider and see the counts, a histogram, the expected churners and (if outcomes are known) a reality check |
| 4. Find customers | Filter by risk, contract, internet and tenure. Sort or search the table, click a row to explain it, and download a CSV. |
| 5. What drives the model | See which factors the model relies on overall |
| 6. Explain one customer | Red bars push risk up and green bars pull it down, compared with a typical customer |
| 7. Suggestions | Offers shown only when the model estimate **and** historical data both point the same way. Set the minimum reduction with the slider. |
| 8. Segment view | See which offers apply to the most flagged customers, to help choose experiments |
| 9. Limitations | A reminder of what not to conclude |

Low-risk customers get the message *"no retention spend suggested"*. The
model's what-if may show a small improvement for them, but the effort is
usually better spent on higher-risk customers.

---

## 9. Using your own customer data

### File requirements

- **CSV, UTF-8 encoded** (Excel: *Save As → CSV UTF-8*), up to 200,000 rows.
- **Exact column names** as in the original dataset. Use **Download example
  CSV** in the app as a template.
- `customerID` is optional but recommended, and IDs must be unique. Without
  it, rows are labelled `row_1`, `row_2`, and so on.
- `Churn` is optional. If you include it (Yes/No), the app compares the
  predictions with what really happened.
- Extra columns are ignored, and column order doesn't matter.
- `TotalCharges` may be blank for brand-new customers (tenure 0).
- Every category must use one of the known values, for example `Contract` =
  `Month-to-month`, `One year` or `Two year`. The app lists all allowed values
  under **Expected CSV format**.

### What happens with a bad file

The **whole file is rejected** with a list of every problem, and nothing is
scored. This is deliberate: partially scored files lead to silent mistakes.
Typical causes are a renamed column, a new product name (such as a new
contract type), a negative number, text in a number column, or duplicate IDs.

### Will the model work on *my* customers?

Only if they resemble the training data. Before relying on it:

1. Score a past period where you **already know who churned** (include the
   `Churn` column).
2. Compare predicted and actual churn in the app's **reality check** table.
3. If the actual churn rates in the Low/Medium/High groups are far from the
   ones in section 5, **do not use the scores**. Ask a data scientist to
   retrain on your own data (section 11).

---

## 10. Privacy, security and responsible use

### Your data stays on your computer

- The app runs **locally**. By default, marimo only accepts connections from
  your own computer (`127.0.0.1`). The project's code does not send customer
  data anywhere.
- Uploaded files are read in memory for scoring. TelcoGuard does not save
  them to disk.
- **Downloads contain customer-level data**: IDs, all account attributes and
  risk scores. Store and share them according to your organisation's
  data-protection rules, and delete them when you no longer need them.
- **Don't expose the app on a network** (for example with `--host 0.0.0.0`)
  without enabling authentication (`--token`) and getting approval from your
  IT/security team.

### Fairness

The model uses `gender` and `SeniorCitizen` as inputs. Before using scores in
decisions that affect customers:

- Check whether flag rates, offers and outcomes differ unfairly across groups.
- Consider whether these attributes should be removed. Your legal or
  compliance team may require it, depending on your jurisdiction and use case.
- No fairness audit has been done yet.

### Ground rules

- ✅ Use scores to **prioritise helpful outreach**: offers, support calls,
  service checks.
- ✅ Keep a **human in the loop** for who is contacted and what is offered.
- ❌ Don't use scores to **penalise** customers, for example with worse prices,
  reduced service or credit decisions.
- ❌ Don't tell customers they are "predicted to leave", or present scores as
  facts about individuals.

---

## 11. Keeping the model healthy

Models get worse as customers, prices and competitors change.

**Retrain when** any of these happen:

- Actual churn in the High group drops well below what section 5 shows, or
  churn in the Low group rises well above it.
- You launch new products or contract types, change prices significantly,
  or enter a new market.
- New CSV files start getting rejected for unknown values. That means your
  data has new categories the model has never seen.
- A fixed schedule (for example every quarter) comes round, if you have
  fresh labelled data.

**How to retrain (data scientist):**

1. Replace or extend `data/Telco-Customer-Churn.csv` with new labelled data
   in the same format. If new categories exist, update `CATEGORY_LEVELS` in
   `src/config.py` first.
2. Run notebooks 01 → 05 in order. Notebook 01 shows whether the data
   changed; 02 makes a new split; 03 retrains; 04 re-evaluates and picks the
   threshold.
3. Run the tests:
   ```bash
   .venv/Scripts/python.exe -m pytest
   ```
   Some tests check facts about the *current* dataset, such as 7,043 rows and
   11 blank charges, so expect to update those for new data.
4. Compare the new Notebook 04 results with the old ones before switching to
   the new model.

The notebooks also stop with a clear message if model files and data don't
match, for example a model trained on an old split.

---

## 12. Frequently asked questions

**Why does the app flag so many customers who don't churn?**
That's the trade-off chosen at threshold 0.30: catching about 78% of
churners means about half of the flags are false alarms. Raise the
threshold, or use only the High group, if contacts are expensive. Lower it if
missing a churner is very costly.

**Why is this customer high risk?**
Click their row and see section 6 for that customer's specific reasons. As a
group, the 187 High-risk test customers share a clear profile:

- all are on month-to-month contracts
- median tenure is 3 months, compared with 47 months for Low-risk customers
- 88% have fibre internet
- 98% have no online security and 94% no tech support
- 75% pay by electronic check

This describes who the model flags, not why those customers leave.

**Can I trust the "model risk after" number for an offer?**
Treat it as a direction, not a forecast. It assumes nothing else changes,
including price, and it reflects patterns rather than causes. Measure real
effects with a control group (section 7).

**Why are "All customers" scores marked as optimistic?**
The model learned from 80% of those customers, so it scores them better than
it would score new ones. Use **Test customers** to judge accuracy.

**Which model is used, and why not the most accurate one?**
Histogram Gradient Boosting, chosen *before* the test evaluation because it
did best in cross-validation. All three models scored almost the same.
Picking the test-set winner would have made the reported accuracy too
optimistic.

**The app says a notebook must be run first. What do I do?**
Run the notebook it names, or ask whoever maintains the project. See the
README's troubleshooting table.

**Can I use it for a different industry?**
Not as-is. The columns and categories are specific to this telecom dataset.
The *workflow* in the notebooks is reusable, but the model would need to be
rebuilt.

---

## 13. Summary: the five things to remember

1. **Prioritise, don't predict individuals.** Scores rank customers; they
   don't decide anyone's fate.
2. **About half of flagged customers would have stayed anyway.** Budget for
   that.
3. **Suggestions are hypotheses.** Always test offers against a control group.
4. **Check the model on your own data** before relying on it, and keep checking.
5. **Handle downloads as personal data**, and keep a human in the loop.
