# Mathematics of the Three Candidate Predictors

This document derives the equations behind the three candidate models
(scikit-learn gradient boosting, XGBoost, LightGBM) and states exactly where
each one departs from the others. Every closed-form result that can be tested
numerically is checked against the real libraries on the real data by
`src/evmlops/verification.py` (see Section 7). Those checks run at the start of
every experiment and are logged to MLflow.

Equations use LaTeX Trial. 

---

## 0. Problem and notation

Each observation is one trip segment. Let $x_i \in \mathcal{X}$ be the input
vector (numeric and categorical trip features) and $y_i > 0$ the observed
consumption in kWh per 100 km, for $i = 1, \dots, n$.

**Target identity and leakage.** In this dataset the target is a deterministic
function of the energy used $E_i$ and the distance $d_i$:

$$
y_i = 100\,\frac{E_i}{d_i}.
$$

This was verified to within rounding error (maximum absolute difference
$5 \times 10^{-4}$). Using $E_i$ as a feature would make the task trivial, so it
is excluded. The end-of-trip battery level is also excluded because it is only
observed after the energy has been consumed.

**Tree.** A regression tree with $T$ leaves is a function

$$
f(x) = w_{q(x)}, \qquad q : \mathcal{X} \to \{1, \dots, T\}, \quad w \in \mathbb{R}^T,
$$

where $q$ routes an input to a leaf and $w_j$ is the value stored in leaf $j$.
Let $I_j = \{ i : q(x_i) = j \}$ be the index set of leaf $j$ and $n_j = |I_j|$.

**Additive model.** All three libraries fit

$$
F_M(x) = F_0 + \nu \sum_{m=1}^{M} f_m(x),
$$

with initial value $F_0$, learning rate (shrinkage) $\nu \in (0, 1]$ and $M$
trees.

**Loss.** Squared error, written with the convention that makes the derivatives
simple:

$$
\ell(y, \hat y) = \tfrac{1}{2} (y - \hat y)^2,
\qquad
g_i = \frac{\partial \ell}{\partial \hat y_i} = \hat y_i - y_i,
\qquad
h_i = \frac{\partial^2 \ell}{\partial \hat y_i^2} = 1.
$$

Here $g_i$ and $h_i$ are the per-observation gradient and Hessian of the loss
evaluated at the current prediction $\hat y_i = F_{m-1}(x_i)$. All three
libraries use exactly these values for the squared-error objective.

---

## 1. Gradient boosting as functional gradient descent (scikit-learn)

Friedman's gradient boosting treats the empirical risk
$R(F) = \sum_i \ell(y_i, F(x_i))$ as a function of the vector of fitted values
$(F(x_1), \dots, F(x_n))$ and descends it one tree at a time.

### 1.1 Initialisation

$$
F_0 = \arg\min_{c} \sum_{i=1}^{n} \ell(y_i, c).
$$

Setting the derivative to zero gives $\sum_i (c - y_i) = 0$, so
$F_0 = \bar y$, the sample mean.

### 1.2 Pseudo-residuals

At stage $m$ the negative gradient of the risk with respect to each fitted value
is

$$
r_{im} = -\left. \frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)} \right|_{F = F_{m-1}}
       = y_i - F_{m-1}(x_i).
$$

For squared error the pseudo-residual is the ordinary residual.

### 1.3 Fit a tree to the residuals

A tree $h_m$ with leaf regions $R_{jm}$ is grown to approximate $r_{im}$ by least
squares. For a node with residuals $r_i$, a split into left and right children
of sizes $n_L$ and $n_R$ ($n = n_L + n_R$) reduces the sum of squared errors by

$$
\Delta \mathrm{SSE}
= \mathrm{SSE}_{\text{parent}} - \mathrm{SSE}_L - \mathrm{SSE}_R
= n_L (\bar r_L - \bar r)^2 + n_R (\bar r_R - \bar r)^2 .
$$

Because $\bar r = (n_L \bar r_L + n_R \bar r_R)/n$, we have
$\bar r_L - \bar r = \tfrac{n_R}{n}(\bar r_L - \bar r_R)$ and
$\bar r_R - \bar r = -\tfrac{n_L}{n}(\bar r_L - \bar r_R)$. Substituting,

$$
\Delta \mathrm{SSE}
= \frac{n_L n_R^2 + n_R n_L^2}{n^2} (\bar r_L - \bar r_R)^2
= \frac{n_L n_R}{n_L + n_R} (\bar r_L - \bar r_R)^2 .
$$

This is Friedman's improvement criterion (`criterion="friedman_mse"`, the
scikit-learn default). The greedy tree builder picks, at every node, the feature
and threshold that maximise it.

### 1.4 Leaf values by line search

Given the regions, the value $\gamma_{jm}$ of each leaf is chosen to minimise the
loss within that leaf:

$$
\gamma_{jm} = \arg\min_{\gamma} \sum_{i \in R_{jm}} \ell\bigl(y_i, F_{m-1}(x_i) + \gamma\bigr).
$$

Setting the derivative to zero, $\sum_{i \in R_{jm}} \bigl(F_{m-1}(x_i) + \gamma - y_i\bigr) = 0$, yields

$$
\gamma_{jm} = \frac{1}{n_{jm}} \sum_{i \in R_{jm}} r_{im} = \bar r_{jm},
$$

the mean residual in the leaf.

### 1.5 Shrunken update

$$
F_m(x) = F_{m-1}(x) + \nu \sum_{j} \gamma_{jm}\, \mathbb{1}[x \in R_{jm}].
$$

**Closed form for the first stage.** After one tree,
$F_1(x_i) = \bar y + \nu\, \bar r_{j(i)}$ with $r_i = y_i - \bar y$. This is tested
against `GradientBoostingRegressor` in `check_sklearn_first_stage`; the
maximum discrepancy on the training data is of the order of $10^{-15}$.

### 1.6 Stochastic variants used in the search

Each stage fits its tree on a random fraction `subsample` of the rows, drawn
without replacement, and considers only a random fraction `max_features` of the
columns at each split. Both reduce the correlation between successive trees,
which lowers variance at the cost of slightly higher bias.

---

## 2. XGBoost: a regularised second-order objective

### 2.1 Objective

XGBoost minimises a regularised objective over the whole ensemble:

$$
\mathcal{L} = \sum_{i=1}^{n} \ell(y_i, \hat y_i) + \sum_{k=1}^{M} \Omega(f_k),
\qquad
\Omega(f) = \gamma\, T + \tfrac{1}{2} \lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j| .
$$

The three penalties are the cost per leaf ($\gamma$), an L2 penalty on leaf
values ($\lambda$) and an L1 penalty ($\alpha$).

### 2.2 Second-order expansion

Boosting adds one tree at a time. With $\hat y_i^{(t-1)}$ fixed, the objective at
round $t$ is

$$
\mathcal{L}^{(t)} = \sum_i \ell\bigl(y_i, \hat y_i^{(t-1)} + f_t(x_i)\bigr) + \Omega(f_t).
$$

A second-order Taylor expansion of the loss around $\hat y_i^{(t-1)}$, dropping
terms that do not depend on $f_t$, gives

$$
\mathcal{L}^{(t)} \approx \sum_i \Bigl[ g_i f_t(x_i) + \tfrac{1}{2} h_i f_t(x_i)^2 \Bigr] + \Omega(f_t).
$$

For squared error the expansion is exact, because the loss is quadratic.

### 2.3 Optimal leaf weights

Group the sum by leaf. With $G_j = \sum_{i \in I_j} g_i$ and
$H_j = \sum_{i \in I_j} h_i$, and taking $\alpha = 0$ first,

$$
\mathcal{L}^{(t)} = \sum_{j=1}^{T} \Bigl[ G_j w_j + \tfrac{1}{2} (H_j + \lambda) w_j^2 \Bigr] + \gamma T .
$$

Each leaf is an independent quadratic in $w_j$. Setting the derivative to zero,

$$
G_j + (H_j + \lambda) w_j = 0
\quad \Longrightarrow \quad
\boxed{\, w_j^{*} = -\frac{G_j}{H_j + \lambda} \,}
$$

Substituting back, the best objective for a fixed tree structure is

$$
\mathcal{L}^{(t)*} = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T .
$$

**With the L1 term.** Minimising $G w + \tfrac12 (H + \lambda) w^2 + \alpha |w|$ by
subgradient calculus gives soft-thresholding:

$$
w^{*} = -\frac{\operatorname{sign}(G)\, \max(|G| - \alpha,\, 0)}{H + \lambda}.
$$

Leaves with a small total gradient are therefore set exactly to zero.

### 2.4 Split gain

Comparing the objective of a node before and after splitting it into left and
right children (with $G = G_L + G_R$ and $H = H_L + H_R$), and accounting for
the extra leaf, gives the gain of a candidate split:

$$
\mathcal{L}_{\text{split}} = \frac{1}{2} \left[
\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}
\right] - \gamma .
$$

A split is only worthwhile when this is positive, so $\gamma$ acts as a
minimum-gain threshold that prunes weak splits.

**Convention note, verified numerically.** The `Gain` value that XGBoost reports
for a split equals the bracketed term without the factor $\tfrac12$ and
before subtracting $\gamma$. On this dataset the closed form gives 187773.002
and `trees_to_dataframe()` reports 187773.0 for the root split of a depth-one
exact-greedy tree.

### 2.5 Specialisation to squared error

With $g_i = -r_i$ and $h_i = 1$ (residuals taken against the current prediction),
$G_j = -\sum_{I_j} r_i$ and $H_j = n_j$, so

$$
w_j^{*} = \frac{\sum_{i \in I_j} r_i}{n_j + \lambda}
        = \frac{n_j}{n_j + \lambda}\, \bar r_j .
$$

This is the mean-residual leaf value of Section 1.4 multiplied by the shrinkage
factor $n_j / (n_j + \lambda) < 1$. The factor tends to one for large leaves and
pulls small, noisy leaves toward zero. As $\lambda \to 0$ XGBoost reduces to
Friedman's gradient boosting; the practical differences are this regulariser,
the explicit leaf penalty $\gamma$, and the use of curvature $h_i$, which
matters for losses other than squared error.

After one tree the prediction is
$\hat y_i = F_0 + \nu \sum_{I_j} r / (n_j + \lambda)$, which is tested against
`XGBRegressor` in `check_xgboost_first_stage` (maximum discrepancy about
$10^{-6}$, the single-precision limit).

### 2.6 Missing values: learned default directions

For a candidate threshold on a feature, XGBoost evaluates the gain twice with
all missing-value rows sent left and then right, and keeps the better one:

$$
\text{gain}(s) = \max\bigl(\text{gain}_{\text{missing} \to L}(s),\; \text{gain}_{\text{missing} \to R}(s)\bigr).
$$

The winning side becomes the default direction stored in the tree. The data
used here has missing values in nine columns, so this matters in practice and is
why the XGBoost and LightGBM pipelines receive the raw NaNs.

### 2.7 Histogram split finding

With `tree_method="hist"` each feature is bucketed into at most `max_bin` bins
(256 by default). For each node the algorithm accumulates $(G, H)$ per bin and
scans the cumulative sums, so the candidate thresholds are the bin edges and the
scan costs $O(\text{bins})$ per feature instead of $O(n)$.

### 2.8 Categorical features

With `enable_categorical=True` categorical columns are split natively. Features
with few categories are split one category versus the rest; for larger
cardinality, categories are ordered by a gradient statistic and a partition is
chosen from that order (see Fisher's result in Section 3.5). The cutoff between
the two is controlled by `max_cat_to_onehot`.

---

## 3. LightGBM: same objective, cheaper and more aggressive search

LightGBM optimises the same second-order objective. For a leaf the optimal
output and the split gain are those of Sections 2.3 and 2.4,

$$
w_j^{*} = -\frac{G_j}{H_j + \lambda},
\qquad
\text{Gain} = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda},
$$

and a split is accepted when the gain exceeds `min_split_gain`. The
implementation reports the gain without the $\tfrac12$ factor. This was verified
numerically: at LightGBM's chosen root split the closed form gives 187773.002
and the library reports 187773.0 (`check_lightgbm_split_gain`). Interestingly,
both libraries chose the same root split on this data.

The first-stage leaf value, including LightGBM's initialisation at the label
mean, matches $\bar y + \nu \sum_{I_j} r_i / (n_j + \lambda)$ (maximum
discrepancy about $7 \times 10^{-8}$, `check_lightgbm_first_stage`).

The differences from XGBoost lie in how trees are grown and how splits are
found.

### 3.1 Leaf-wise (best-first) growth

Depth-wise growth splits every node at a level before going deeper. LightGBM
instead keeps all current leaves in a priority queue and, at each step, splits
the one with the largest available gain:

$$
\ell^{\star} = \arg\max_{\ell \in \text{leaves}} \; \max_{s} \; \text{Gain}(\ell, s).
$$

Growth stops at `num_leaves` $= T$ leaves. For the same number of leaves this
greedy rule usually reaches a lower training loss than level-wise growth,
because it spends splits where the loss falls most. The trade-off is that it
can produce deep, unbalanced trees that overfit small leaves, so
`min_child_samples` (a minimum $n_j$) and `num_leaves` are the key controls.
Note this is a greedy heuristic, not a guarantee of a better tree.

### 3.2 Histogram algorithm and the subtraction trick

Features are discretised once into at most 255 bins. A histogram of
$(\sum g, \sum h, \text{count})$ per bin is built for a node in
$O(n_{\text{node}}\, p)$ and the best split is found by scanning bins in
$O(\text{bins}\cdot p)$. When a node splits, only the histogram of the smaller
child is computed from data; the sibling follows by subtraction,

$$
H_{\text{sibling}} = H_{\text{parent}} - H_{\text{smaller child}},
$$

which roughly halves the histogram work at every level.

### 3.3 Gradient-based One-Side Sampling (GOSS)

GOSS is an optional sampling strategy (`data_sample_strategy="goss"`). It is
documented here because it is the main algorithmic idea of the LightGBM paper,
but the candidate in this module uses the default `gbdt` with bagging
(`subsample` with `subsample_freq=1`), so GOSS is not active.

For a node $O$ with $n_O$ rows, the variance gain of splitting feature $j$ at
$d$ (unit Hessians, $\lambda = 0$) is

$$
V_{j|O}(d) = \frac{1}{n_O}\left(
\frac{\bigl(\sum_{x_{ij} \le d} g_i\bigr)^2}{n^{j}_{l|O}(d)} +
\frac{\bigl(\sum_{x_{ij} > d} g_i\bigr)^2}{n^{j}_{r|O}(d)}
\right),
$$

which is the gain of Section 2.4 up to the parent term, which does not depend on
$d$. Rows with large $|g_i|$ carry the most information about the split. GOSS
keeps the top $a$ fraction by $|g_i|$ (set $A$), samples a fraction $b$ of all
rows uniformly from the remainder $A^c$ (set $B$), and estimates the gain as

$$
\tilde V_{j}(d) = \frac{1}{n}\left(
\frac{\bigl(\sum_{A_l} g_i + \frac{1 - a}{b} \sum_{B_l} g_i\bigr)^2}{n_l^{j}(d)} +
\frac{\bigl(\sum_{A_r} g_i + \frac{1 - a}{b} \sum_{B_r} g_i\bigr)^2}{n_r^{j}(d)}
\right).
$$

The factor $\frac{1-a}{b}$ makes the sampled sum an unbiased estimate. Because
$B$ is a uniform sample of $b\,n$ of the $(1-a)\,n$ rows in $A^c$,

$$
\mathbb{E}\Bigl[\sum_{i \in B} g_i\Bigr] = \frac{b\,n}{(1-a)\,n} \sum_{i \in A^c} g_i
\quad \Longrightarrow \quad
\mathbb{E}\Bigl[\tfrac{1-a}{b} \sum_{i \in B} g_i\Bigr] = \sum_{i \in A^c} g_i .
$$

### 3.4 Exclusive Feature Bundling (EFB)

EFB merges mutually exclusive sparse features into single columns, using a
greedy graph-colouring heuristic because optimal bundling is NP-hard. It targets
high-dimensional sparse data. This dataset has 22 mostly dense features, so EFB
has no material effect here.

### 3.5 Native categorical splits and Fisher's result

For squared error, the optimal binary partition of the $K$ levels of a
categorical feature does not require searching all $2^{K-1} - 1$ partitions.
Fisher (1958) showed that sorting the categories by their mean response and
considering only contiguous cuts of that order is optimal. LightGBM applies this
to second-order statistics: categories are sorted by

$$
\frac{\sum_{i \in c} g_i}{\sum_{i \in c} h_i + \text{cat\_smooth}}
$$

and the best cut of that order is chosen, reducing the search to $K - 1$
candidates. The smoothing term stabilises rare categories.

### 3.6 Missing values

LightGBM also learns which side missing values take at each split, by the same
two-way gain comparison as in Section 2.6.

---

## 4. Mapping the equations to hyperparameters

| Symbol | Meaning | scikit-learn | XGBoost | LightGBM |
|---|---|---|---|---|
| $\nu$ | shrinkage | `learning_rate` | `learning_rate` | `learning_rate` |
| $M$ | number of trees | `n_estimators` | `n_estimators` | `n_estimators` |
| $T$ | leaves per tree | via `max_depth` ($T \le 2^{D}$) | via `max_depth` ($T \le 2^{D}$) | `num_leaves` |
| $\lambda$ | L2 on leaf values | none | `reg_lambda` | `reg_lambda` |
| $\alpha$ | L1 on leaf values | none | `reg_alpha` | `reg_alpha` |
| $\gamma$ | minimum gain / leaf cost | none | `gamma` | `min_split_gain` |
| minimum $H_j$ | smallest allowed leaf | `min_samples_leaf` (count) | `min_child_weight` (sum of $h$) | `min_child_samples` (count) |
| row sampling | stochastic boosting | `subsample` | `subsample` | `subsample` with `subsample_freq=1` |
| column sampling | | `max_features` | `colsample_bytree` | `colsample_bytree` |

For squared error $h_i = 1$, so XGBoost's `min_child_weight` is a minimum number
of rows, matching the count-based parameters of the other two libraries.

scikit-learn's `GradientBoostingRegressor` has no $\lambda$, $\alpha$ or
$\gamma$. Its regularisation comes from shrinkage, depth, leaf size and
subsampling.

---

## 5. Why shrinkage and regularisation help

Writing the leaf value as $\nu\, w_j^{*}$ shows two separate brakes. The
learning rate $\nu$ scales every tree's contribution, so more trees are needed
but the path of $F_m$ through function space is smoother (a standard
bias-variance trade-off: small $\nu$ with large $M$ usually generalises better
than large $\nu$ with small $M$). The factor $n_j / (n_j + \lambda)$ from
Section 2.5 acts per leaf and is strongest where leaves are small, which is
precisely where noise dominates.

`diagnostics.staged_rmse` traces the test error of the partial sums $F_m$ for
$m = 1, \dots, M$, so this trade-off is visible in the logged
`staged_error.png`.

---

## 6. Evaluation mathematics

### 6.1 Point metrics

For test predictions $\hat y_i$ over $n$ rows, with $e_i = \hat y_i - y_i$:

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_i e_i^2}, \quad
\mathrm{MAE} = \frac{1}{n}\sum_i |e_i|, \quad
R^2 = 1 - \frac{\sum_i e_i^2}{\sum_i (y_i - \bar y)^2},
$$

$$
\mathrm{MAPE} = \frac{100}{n}\sum_i \frac{|e_i|}{|y_i|}, \quad
\text{bias} = \frac{1}{n}\sum_i e_i .
$$

Positive bias means the model over-predicts on average. A mean-predictor
baseline has $R^2$ close to zero on held-out data and gives a reference for
how much signal each model captured.

### 6.2 Grouped validation

Trips from the same vehicle share characteristics (for example tyre pressure
drifts slowly and acts almost like a vehicle identifier). If rows from one
vehicle appear in both training and validation, a flexible model can partly
memorise the vehicle and the error looks smaller than it will be on a new
vehicle. All splits are therefore made by vehicle:

* the held-out test set is a `GroupShuffleSplit` by `vehicle_id`;
* hyperparameter search uses `GroupKFold` by `vehicle_id` within the training
  vehicles;
* model selection uses the grouped cross-validation RMSE only. The test set is
  used once per model to report performance and never to choose among models.

### 6.3 Cluster bootstrap for uncertainty

Let the test set contain $G$ vehicles, with $n_g$ rows and total squared error
$S_g = \sum_{i \in g} e_i^2$ for vehicle $g$. A bootstrap replicate $b$ draws
vehicle counts

$$
c^{(b)} = (c^{(b)}_1, \dots, c^{(b)}_G) \sim \mathrm{Multinomial}\bigl(G;\, 1/G, \dots, 1/G\bigr),
$$

and recomputes the metric from per-vehicle sufficient statistics,

$$
\mathrm{RMSE}^{(b)} = \sqrt{\frac{\sum_g c^{(b)}_g S_g}{\sum_g c^{(b)}_g n_g}} .
$$

The 95 percent interval is the 2.5th to 97.5th percentile of the replicates.
Resampling whole vehicles respects the dependence between a vehicle's rows,
which a row-level bootstrap would ignore (understating uncertainty).

For model pairs $A$ and $B$ the same replicates are used for both models, so the
difference

$$
\Delta^{(b)} = \mathrm{RMSE}_A^{(b)} - \mathrm{RMSE}_B^{(b)}
$$

is a paired comparison. The module logs its mean, percentile interval and the
share of replicates with $\Delta^{(b)} < 0$. An interval that contains zero means
the data cannot separate the two models.

### 6.4 Permutation importance

For input column $k$, shuffle its values across the test rows and measure the
increase in RMSE,

$$
\mathrm{PI}_k = \mathrm{RMSE}\bigl(\text{shuffled } k\bigr) - \mathrm{RMSE}\bigl(\text{original}\bigr),
$$

averaged over repeats. It is computed on the raw input columns of the full
pipeline, so all three models are measured in the same units regardless of
their internal encodings. It is a model-agnostic complement to the
library-native gain importances, which are not comparable across libraries.

---

## 7. Numerical verification of the derivations

`evmlops.verification.run_derivation_checks` fits single-tree models on the
training data and compares each library to the closed form. Results on this
dataset (learning rate 0.3, $\lambda = 5$ where applicable):

| Check | Library | Closed form | Max abs. discrepancy |
|---|---|---|---|
| First-stage predictions | scikit-learn | $\bar y + \nu\, \bar r_j$ | $\approx 4 \times 10^{-15}$ |
| First-stage predictions | XGBoost | $\bar y + \nu \sum r / (n_j + \lambda)$ | $\approx 1 \times 10^{-6}$ |
| First-stage predictions | LightGBM | $\bar y + \nu \sum r / (n_j + \lambda)$ | $\approx 7 \times 10^{-8}$ |
| Root split gain | XGBoost | $\sum \frac{G^2}{H + \lambda}$ bracket | 187773.002 vs 187773.0 |
| Root split gain | LightGBM | $\sum \frac{G^2}{H + \lambda}$ bracket | 187773.002 vs 187773.0 |

The same tests also run in `tests/test_verification.py`.

---

## References

1. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting
   machine. Available in *Annals of Statistics*, 29(5), 1189-1232.
2. Friedman, J. H. (2002). Stochastic gradient boosting. Available in *Computational
   Statistics and Data Analysis*, 38(4), 367-378.
3. Chen, T. and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. Available in
   *Proceedings of KDD*, 785-794.
4. Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision 
   tree. Available in *Advances in Neural Information Processing Systems*, 30.
5. Fisher, W. D. (1958). On grouping for maximum homogeneity. Available in *Journal of the
   American Statistical Association*, 53(284), 789-798.
