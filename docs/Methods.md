# Beta-Binomial 矩估计（估计甲基化 Beta 先验的 α, β）

## 1. 模型与参数化

每个位点 $i$ 有覆盖 $n_i$ 、甲基化计数 $k_i$ 。假设位点真实甲基化率 $p_i \sim \text{Beta}(\alpha, \beta)$ ，观测 $k_i \mid p_i \sim \text{Binomial}(n_i, p_i)$ 。

用两个更好解释的参数：

$$
\mu = \frac{\alpha}{\alpha + \beta}\ (\text{均值}), \qquad
\kappa = \alpha + \beta\ (\text{浓度}), \qquad
\rho = \frac{1}{\kappa + 1}\ (\text{位点内相关 / 过离散})
$$

反过来：

$$
\alpha = \mu\kappa, \qquad \beta = (1-\mu)\kappa, \qquad \kappa = \frac{1-\rho}{\rho}
$$

## 2. Beta-Binomial 的边际矩（关键）

$$
E[k_i] = n_i \mu
$$

$$
\operatorname{Var}(k_i) = n_i\, \mu(1-\mu)\, \bigl[\, 1 + (n_i - 1)\rho \,\bigr]
$$

比纯 Binomial 多了过离散因子 $[1 + (n_i - 1)\rho]$ ——这正是"用 cov 校正"的来源。 $\rho = 0$ 退化为 Binomial。

## 3. 矩估计（可处理不等覆盖，推荐）

**第一矩 → μ**（池化，即"全局 mc/cov"）：

$$
\hat\mu = \frac{\sum_i k_i}{\sum_i n_i}
$$

**第二矩 → ρ**：构造一个 Pearson 型过离散统计量

$$
X = \sum_i \frac{(k_i - n_i \hat\mu)^2}{n_i\, \hat\mu(1-\hat\mu)}
$$

因为

$$
E\!\left[\frac{(k_i - n_i \mu)^2}{n_i \mu(1-\mu)}\right] = 1 + (n_i - 1)\rho
$$

所以 $E[X] = N + \rho \sum_i (n_i - 1)$ （ $N$ = 位点数）。令 $X = E[X]$ 解出

$$
\boxed{\ \hat\rho = \frac{X - N}{\sum_i (n_i - 1)}\ }
$$

（更严的自由度修正把分子 $N$ 换成 $N-1$ ，即 Tarone 1979 的形式；数据量大时差别可忽略。）

**转成 α, β**：

$$
\hat\kappa = \frac{1 - \hat\rho}{\hat\rho}, \qquad
\hat\alpha = \hat\mu\, \hat\kappa, \qquad
\hat\beta = (1 - \hat\mu)\, \hat\kappa
$$

## 4. 等价的"频率方差"直觉版（覆盖近似相等时）

设 $f_i = k_i / n_i$ ，则

$$
\operatorname{Var}(f_i) = \mu(1-\mu)\left[\frac{1}{n_i} + \left(1 - \frac{1}{n_i}\right)\rho\right]
$$

若各位点覆盖 $\approx \bar n$ ，用观测方差 $s^2 = \operatorname{Var}(f_i)$ 解：

$$
\hat\rho = \frac{\dfrac{s^2}{\mu(1-\mu)} - \dfrac{1}{\bar n}}{1 - \dfrac{1}{\bar n}}
$$

这就明确显示了"从率方差里**扣掉 binomial 抽样项 $1/\bar n$**"再得到真实的 Beta 离散——这是它比 ALLCools 纯 Beta-MoM（不扣抽样噪声）更准的地方。覆盖差异大时应该用第 3 节的 $X$ 版本（对每个 $n_i$ 加权），不要用这个近似。

## 5. 边界与实现要点

- **裁剪** $\hat\rho \in (\epsilon,\ 1-\epsilon)$ ：
  - $\hat\rho \le 0$ （欠离散 / 被抽样噪声压过）→ 令 $\rho \to \epsilon$ ，即 $\kappa$ 很大（强 prior）；
  - $\hat\rho \to 1$ → $\kappa \to 0$ （几乎无收缩）。
- $\hat\mu$ 也裁剪到 $(\epsilon, 1-\epsilon)$ ，避免 $\log$ 发散。
- **过滤 $n_i = 0$**，最好 $n_i \ge 2$ （ $n_i = 1$ 对 $\sum(n_i - 1)$ 无贡献且噪声大）。
- **CpG / CpH 分开各估一套**（背景完全不同）。
- 在**深测序 pseudobulk 参考**上估，而非浅测序 query cell。

## 附：与 ALLCools 的区别

ALLCools（`calculate_posterior_mc_frac`）用的是**纯 Beta 矩估计**：直接对观测率 $f_i = mc/cov$ 求均值 $\mu$ 和方差 $\sigma^2$ ，套用

$$
\alpha = \mu\left[\frac{\mu(1-\mu)}{\sigma^2} - 1\right], \qquad
\beta = \alpha\left(\frac{1}{\mu} - 1\right)
$$

它**没有扣除 binomial 抽样噪声**（ $\sigma^2$ 里混了抽样方差），所以在低覆盖数据上会高估方差、低估浓度 $\kappa$ 。上面的 Beta-Binomial 版本通过第 2 节的过离散因子对覆盖做了校正。

> 说明：当前 `ALLCools/mcds/utilities.py` 里的 `calculate_posterior_mc_frac` 已改为本文第 3 节的 Beta-Binomial 矩估计实现；下方 `calculate_posterior_mc_frac_deprecated` 保留的是旧的纯 Beta-MoM。

## 6. 新方法 vs 旧方法：好在哪里

**结论：新方法（Beta-Binomial 矩估计）在单细胞甲基化这种低/不等覆盖的场景下更准，且不会带来额外的显著计算开销。** 具体优势如下。

### 6.1 正确扣除了 binomial 抽样噪声

观测率 $f_i = k_i/n_i$ 的方差包含两部分：

$$
\operatorname{Var}(f_i) = \underbrace{\mu(1-\mu)\rho}_{\text{生物学真实离散}} + \underbrace{\frac{\mu(1-\mu)(1-\rho)}{n_i}}_{\text{binomial 抽样噪声}}
$$

- **旧方法**把整个 $\operatorname{Var}(f_i)$ 都当作 Beta 离散，于是**高估方差、低估浓度 $\kappa$**——先验被当成"比真实更宽"，收缩太弱。
- **新方法**用第 2 节的过离散因子 $[1+(n_i-1)\rho]$ 显式扣掉了 $1/n_i$ 那一项，得到的是**真实的位点间离散 $\rho$**。

覆盖越低（ $n_i$ 越小），抽样噪声项越大，旧方法的偏差越严重——而单细胞甲基化恰恰是低覆盖场景。

### 6.2 正确处理不等覆盖（按 $n_i$ 加权）

- 旧方法对每个位点的 $f_i$ **等权**求均值/方差：一个 $cov=1$ 的位点（ $f_i$ 只能取 0 或 1，噪声极大）和一个 $cov=100$ 的位点被同等对待。
- 新方法的 $\hat\mu=\sum k_i/\sum n_i$ 是**按覆盖加权**的池化估计， $X$ 统计量对每个 $n_i$ 单独加权，天然地让高覆盖位点贡献更多、低覆盖位点贡献更少。

### 6.3 均值估计更稳、无 $0/0$ 病态

- 旧方法先对每个位点算 $f_i=mc/cov$ 再平均， $cov=0$ 会产生 `NaN`、 $cov$ 很小时 $f_i$ 抖动剧烈。
- 新方法先求和再相除（ $\sum k_i / \sum n_i$ ），对 $cov=0$ 的位点自然免疫，估计更平滑。

### 6.4 参数有明确、可解释的物理含义

新方法把先验拆成 $\mu$ （背景甲基化水平）和 $\rho$ （位点内相关 / 过离散），二者都可解释、可分别裁剪；CpG / CpH 背景差异也能自然体现在各自的 $\rho$ 上。

### 6.5 下游后验一致

先验 $\alpha,\beta$ 更准 $\Rightarrow$ 共轭后验 $\text{Beta}(\alpha+k_i,\ \beta+n_i-k_i)$ 的**收缩强度更合理**：低覆盖位点向背景 $\mu$ 收缩得当，高覆盖位点保留自身信息。后验均值 `post_frac` 和后验标准差 `post_sigma` 的公式不变，但因为 $\alpha,\beta$ 更准，数值也更可靠。

### 6.6 什么时候两者差别不大

当所有位点覆盖都很高且大致相等（ $n_i \gg 1$ 、 $\bar n$ 接近）时，抽样噪声项 $1/n_i \to 0$ ，两种方法趋于一致。**差别主要出现在低覆盖、覆盖高度不均的数据上**——也就是单细胞甲基化的典型情形，因此推荐新方法。

### 局限与注意

- 新方法只估**一套全局 $(\alpha,\beta)$/每个 mc_type**，属于矩估计，不是完整贝叶斯/最大似然；若需要更精细可上 Beta-Binomial MLE，但计算更贵。
- 仍需注意在**深测序 pseudobulk 参考**上估计，而非浅测序 query cell（见第 5 节）。

---

# English Version — Beta-Binomial Method of Moments (estimating the Beta prior α, β for methylation)

## 1. Model and parameterization

Each site $i$ has coverage $n_i$ and methylated count $k_i$. Assume the site's true methylation rate $p_i \sim \text{Beta}(\alpha, \beta)$, and the observation $k_i \mid p_i \sim \text{Binomial}(n_i, p_i)$.

Use two more interpretable parameters:

$$
\mu = \frac{\alpha}{\alpha + \beta}\ (\text{mean}), \qquad
\kappa = \alpha + \beta\ (\text{concentration}), \qquad
\rho = \frac{1}{\kappa + 1}\ (\text{intra-class correlation / over-dispersion})
$$

Inversely:

$$
\alpha = \mu\kappa, \qquad \beta = (1-\mu)\kappa, \qquad \kappa = \frac{1-\rho}{\rho}
$$

## 2. Marginal moments of the Beta-Binomial (the crux)

$$
E[k_i] = n_i \mu
$$

$$
\operatorname{Var}(k_i) = n_i\, \mu(1-\mu)\, \bigl[\, 1 + (n_i - 1)\rho \,\bigr]
$$

Compared with a plain Binomial, there is an extra over-dispersion factor $[1 + (n_i - 1)\rho]$ — this is exactly where the "coverage correction" comes from. $\rho = 0$ degenerates to a Binomial.

## 3. Method of moments (handles unequal coverage, recommended)

**First moment → μ** (pooled, i.e. the "global mc/cov"):

$$
\hat\mu = \frac{\sum_i k_i}{\sum_i n_i}
$$

**Second moment → ρ**: build a Pearson-type over-dispersion statistic

$$
X = \sum_i \frac{(k_i - n_i \hat\mu)^2}{n_i\, \hat\mu(1-\hat\mu)}
$$

Because

$$
E\!\left[\frac{(k_i - n_i \mu)^2}{n_i \mu(1-\mu)}\right] = 1 + (n_i - 1)\rho
$$

we have $E[X] = N + \rho \sum_i (n_i - 1)$ ($N$ = number of sites). Setting $X = E[X]$ and solving:

$$
\boxed{\ \hat\rho = \frac{X - N}{\sum_i (n_i - 1)}\ }
$$

(A stricter degrees-of-freedom correction replaces the numerator $N$ with $N-1$, i.e. the Tarone 1979 form; the difference is negligible for large datasets.)

**Convert to α, β**:

$$
\hat\kappa = \frac{1 - \hat\rho}{\hat\rho}, \qquad
\hat\alpha = \hat\mu\, \hat\kappa, \qquad
\hat\beta = (1 - \hat\mu)\, \hat\kappa
$$

## 4. Equivalent "rate-variance" intuition (when coverage is roughly equal)

Let $f_i = k_i / n_i$, then

$$
\operatorname{Var}(f_i) = \mu(1-\mu)\left[\frac{1}{n_i} + \left(1 - \frac{1}{n_i}\right)\rho\right]
$$

If all sites have coverage $\approx \bar n$, solve using the observed variance $s^2 = \operatorname{Var}(f_i)$:

$$
\hat\rho = \frac{\dfrac{s^2}{\mu(1-\mu)} - \dfrac{1}{\bar n}}{1 - \dfrac{1}{\bar n}}
$$

This makes explicit that we "**subtract the binomial sampling term $1/\bar n$**" from the rate variance to recover the true Beta dispersion — that is where it is more accurate than ALLCools' plain Beta-MoM (which does not remove the sampling noise). When coverage varies a lot, use the $X$ version of Section 3 (which weights each $n_i$), not this approximation.

## 5. Boundary conditions and implementation notes

- **Clamp** $\hat\rho \in (\epsilon,\ 1-\epsilon)$:
  - $\hat\rho \le 0$ (under-dispersion / overwhelmed by sampling noise) → set $\rho \to \epsilon$, i.e. very large $\kappa$ (strong prior);
  - $\hat\rho \to 1$ → $\kappa \to 0$ (almost no shrinkage).
- Also clamp $\hat\mu$ to $(\epsilon, 1-\epsilon)$ to avoid $\log$ divergence.
- **Filter $n_i = 0$**, preferably keep $n_i \ge 2$ ($n_i = 1$ contributes nothing to $\sum(n_i - 1)$ and is noisy).
- **Estimate CpG / CpH separately** (their backgrounds differ completely).
- Estimate on a **deep-sequencing pseudobulk reference**, not on shallow query cells.

## Appendix: Difference from ALLCools

ALLCools (`calculate_posterior_mc_frac`) uses a **plain Beta method of moments**: it takes the mean $\mu$ and variance $\sigma^2$ of the observed rates $f_i = mc/cov$ directly and applies

$$
\alpha = \mu\left[\frac{\mu(1-\mu)}{\sigma^2} - 1\right], \qquad
\beta = \alpha\left(\frac{1}{\mu} - 1\right)
$$

It **does not remove the binomial sampling noise** ($\sigma^2$ mixes in the sampling variance), so on low-coverage data it over-estimates the variance and under-estimates the concentration $\kappa$. The Beta-Binomial version above corrects for coverage via the over-dispersion factor of Section 2.

> Note: the current `calculate_posterior_mc_frac` in `ALLCools/mcds/utilities.py` has been switched to the Beta-Binomial method-of-moments implementation of Section 3; the retained `calculate_posterior_mc_frac_deprecated` is the old plain Beta-MoM.

## 6. New vs. old method: why it is better

**Conclusion: the new method (Beta-Binomial MoM) is more accurate for single-cell methylation, a low / unequal-coverage setting, without adding meaningful compute cost.** The specific advantages follow.

### 6.1 Correctly removes the binomial sampling noise

The variance of the observed rate $f_i = k_i/n_i$ has two parts:

$$
\operatorname{Var}(f_i) = \underbrace{\mu(1-\mu)\rho}_{\text{true biological dispersion}} + \underbrace{\frac{\mu(1-\mu)(1-\rho)}{n_i}}_{\text{binomial sampling noise}}
$$

- **Old method** treats the whole $\operatorname{Var}(f_i)$ as Beta dispersion, so it **over-estimates the variance and under-estimates the concentration $\kappa$** — the prior is treated as "wider than reality" and shrinkage is too weak.
- **New method** uses the over-dispersion factor $[1+(n_i-1)\rho]$ of Section 2 to explicitly subtract the $1/n_i$ term, recovering the **true between-site dispersion $\rho$**.

The lower the coverage (smaller $n_i$), the larger the sampling-noise term, and the worse the old method's bias — and single-cell methylation is exactly the low-coverage regime.

### 6.2 Correctly handles unequal coverage (weighting by $n_i$)

- The old method averages the per-site $f_i$ with **equal weight**: a $cov=1$ site (where $f_i$ can only be 0 or 1, extremely noisy) and a $cov=100$ site are treated equally.
- The new method's $\hat\mu=\sum k_i/\sum n_i$ is a **coverage-weighted** pooled estimate, and the $X$ statistic weights each $n_i$ individually, so high-coverage sites naturally contribute more and low-coverage sites less.

### 6.3 More stable mean, no $0/0$ pathology

- The old method computes $f_i=mc/cov$ per site and then averages; $cov=0$ produces `NaN` and small $cov$ makes $f_i$ jump wildly.
- The new method sums first and divides once ($\sum k_i / \sum n_i$), is naturally immune to $cov=0$ sites, and is smoother.

### 6.4 Parameters have a clear, interpretable meaning

The new method decomposes the prior into $\mu$ (background methylation level) and $\rho$ (intra-class correlation / over-dispersion); both are interpretable and can be clamped separately, and the CpG / CpH background difference shows up naturally in their respective $\rho$.

### 6.5 Consistent downstream posterior

More accurate priors $\alpha,\beta$ $\Rightarrow$ the conjugate posterior $\text{Beta}(\alpha+k_i,\ \beta+n_i-k_i)$ has a **more reasonable shrinkage strength**: low-coverage sites shrink appropriately toward the background $\mu$, high-coverage sites keep their own information. The posterior mean `post_frac` and posterior std `post_sigma` formulas are unchanged, but because $\alpha,\beta$ are more accurate, the numbers are more reliable too.

### 6.6 When the two barely differ

When all sites have high and roughly equal coverage ($n_i \gg 1$, $\bar n$ close), the sampling-noise term $1/n_i \to 0$ and the two methods converge. **The difference mainly appears on low-coverage, highly uneven-coverage data** — i.e. the typical single-cell methylation case — hence the new method is recommended.

### Limitations and caveats

- The new method estimates only **a single global $(\alpha,\beta)$ per mc_type**; it is a moment estimator, not a full Bayesian / maximum-likelihood fit. For more precision use a Beta-Binomial MLE, but it is more expensive.
- Still estimate on a **deep-sequencing pseudobulk reference**, not on shallow query cells (see Section 5).

## 7. In-code use (cytozip)

The same Beta-Binomial MoM is used in two places, with different pooling axes:

- **`cytozip/model.py` → `estimate_beta_prior(mc, cov, ...)`**: per-context prior, pooled over sites (and over all cell types), estimated on the **deep pseudobulk reference** at `fit` time (CpG / CpH each get their own prior). Feeds `estimate_theta`.
- **`cytozip/features.py` → `_compute_beta_params_sparse`**: **per-cell** prior, pooled over that cell's features (sparse-streamed over the CSR mc/cov layers), written to `adata.obs` as `alpha`, `beta`, `prior_mean`, `rho` (only when `score='posterior_frac'`).
  - The exported `rho = 1/(alpha+beta+1)` is coverage-independent, so it doubles as a **per-cell QC handle** (flag degenerate / low-complexity cells), and lets downstream code recover the shrinkage strength $\kappa=(1-\rho)/\rho$.

## 8. 后验甲基化分数（Posterior methylation fraction）

有了每个细胞的 Beta 先验 $\text{Beta}(\alpha,\beta)$ （第 3 节的矩估计）后，某位点/特征观测到 $mc$ 个甲基化、覆盖 $cov$ ，由 Beta–Binomial 的**共轭性**得到该位点真实甲基化率的后验：

$$
p \mid mc, cov \ \sim\ \text{Beta}\bigl(\alpha + mc,\ \beta + cov - mc\bigr)
$$

**后验均值**（即 `posterior_frac`，用作收缩后的甲基化分数）：

$$
\boxed{\ \widehat{p}_{\text{post}} = \frac{mc + \alpha}{cov + \alpha + \beta}\ }
$$

**后验标准差**（可选，`post_sigma`）：

$$
\sigma_{\text{post}} = \sqrt{\dfrac{(\alpha + mc)\,(\beta + cov - mc)}{(\alpha + \beta + cov + 1)\,(\alpha + \beta + cov)^2}}
$$

**直觉**：把后验均值改写成先验均值 $\mu=\alpha/(\alpha+\beta)$ 与观测率 $f=mc/cov$ 的凸组合：

$$
\widehat{p}_{\text{post}} = w\,f + (1-w)\,\mu, \qquad w = \frac{cov}{cov + \alpha + \beta}
$$

覆盖 $cov$ 越低，权重 $w$ 越小，越向背景 $\mu$ 收缩，从而**扣除低覆盖的抽样噪声**；覆盖越高越保留观测本身。这正是它比原始分数 $mc/cov$ 更稳的原因。

**可选的按细胞归一化（ALLCools 口径）**： $\widehat{p}_{\text{post}} / \mu$ ，使 $cov=0$ 的特征归一化率恒为 1（即"无信息"）。

**cytozip 实现要点**（`features.py`）：

- `score='posterior_frac'` 时 `.X` 存**未归一化**的后验均值 $\dfrac{mc+\alpha}{cov+\alpha+\beta}$ （尺度仍在 $[0,1]$ ，与 `frac` 可比）。
- 仅在**覆盖到的条目**（ $cov>0$ ）上计算；未覆盖条目保持隐式 0（与 `frac` 相同的稀疏结构）。
- $\alpha,\beta$ 为**每个细胞一套**（对该细胞的所有特征池化），写入 `adata.obs['alpha','beta','prior_mean','rho']`；退化细胞（ $\alpha/\beta$ 为 NaN）回退到原始分数 $mc/cov$ 。

## 9. 高变特征累加量与离散度（HVF：`hvf_n_cov` / `hvf_sum` / `hvf_sum_sq`、var、dispersion、normalized dispersion）

为了在**不重新读取矩阵**（甚至跨多个文件合并）的前提下选出高变特征（HVF），对每个特征 $j$ 存三个**可加**的累加量。设 $f_{ij}$ 为细胞 $i$ 在特征 $j$ 上的甲基化分数（默认用第 8 节的后验均值，退化细胞回退原始 $mc/cov$ ；也可设 `hvf_frac='raw'` 直接用 $mc/cov$ ），只在覆盖到的细胞（ $cov_{ij}>0$ ）上累加：

$$
\text{hvf\_n\_cov}_j = \sum_i \mathbb{1}(cov_{ij}>0), \qquad
\text{hvf\_sum}_j = \sum_i f_{ij}, \qquad
\text{hvf\_sum\_sq}_j = \sum_i f_{ij}^2
$$

三者都是对细胞求和，**跨细胞、跨数据集可加**（合并时逐特征相加即可），写入 `adata.var`。由它们可重构：

**均值与方差**：

$$
\text{mean}_j = \frac{\text{hvf\_sum}_j}{\text{hvf\_n\_cov}_j}, \qquad
\text{var}_j = \frac{\text{hvf\_sum\_sq}_j}{\text{hvf\_n\_cov}_j} - \text{mean}_j^2
$$

**离散度（dispersion）**：

$$
\text{disp}_j = \frac{\text{var}_j}{\text{mean}_j}
$$

**归一化离散度（normalized dispersion，scanpy 'seurat' 口径）**：把所有特征按 $\text{mean}$ 分到 $K$ 个箱（例如 $K=20$ ），在**每个箱内**对 dispersion 做 z-score，避免只挑到高均值特征：

$$
\text{ndisp}_j = \frac{\text{disp}_j - \mu_{b(j)}}{\sigma_{b(j)}}
$$

其中 $b(j)$ 是特征 $j$ 所在的均值箱， $\mu_{b},\sigma_{b}$ 是该箱内 dispersion 的均值与标准差。最后取 $\text{ndisp}$ 最高的 top-$n$ 个特征作为 HVF。

**为什么只存三个累加量**：`mean/var/dispersion` 都能由 $(\text{hvf\_n\_cov},\text{hvf\_sum},\text{hvf\_sum\_sq})$ 精确重构，而这三者可加 —— 所以合并多个 `.h5ad`（例如 `AnnDataCollection.from_files(..., var_agg='sum')`）后能在**合并后的全体细胞**上正确重算 HVF。**注意**：`mean/var/dispersion` 本身**不可加**，不要直接对它们求和；尤其 `normalized dispersion` 依赖全体特征的分箱，必须**最后一步**统一计算。覆盖细胞数少于阈值（如 `min_cells`）的特征在选择时置为不合格。

---

## 8. Posterior methylation fraction (English)

Given each cell's Beta prior $\text{Beta}(\alpha,\beta)$ (the MoM of Section 3), a site/feature observed with $mc$ methylated calls out of coverage $cov$ has, by Beta–Binomial **conjugacy**, the posterior of the true rate:

$$
p \mid mc, cov \ \sim\ \text{Beta}\bigl(\alpha + mc,\ \beta + cov - mc\bigr)
$$

**Posterior mean** (this is `posterior_frac`, the shrunk methylation fraction):

$$
\boxed{\ \widehat{p}_{\text{post}} = \frac{mc + \alpha}{cov + \alpha + \beta}\ }
$$

**Posterior std** (optional, `post_sigma`):

$$
\sigma_{\text{post}} = \sqrt{\dfrac{(\alpha + mc)\,(\beta + cov - mc)}{(\alpha + \beta + cov + 1)\,(\alpha + \beta + cov)^2}}
$$

**Intuition**: rewrite the posterior mean as a convex combination of the prior mean $\mu=\alpha/(\alpha+\beta)$ and the observed rate $f=mc/cov$:

$$
\widehat{p}_{\text{post}} = w\,f + (1-w)\,\mu, \qquad w = \frac{cov}{cov + \alpha + \beta}
$$

The lower the coverage $cov$, the smaller the weight $w$ and the stronger the shrinkage toward the background $\mu$, which **removes the low-coverage sampling noise**; high-coverage sites keep their own signal. That is why it is more stable than the raw fraction $mc/cov$.

**Optional per-cell normalization (ALLCools convention)**: $\widehat{p}_{\text{post}} / \mu$, which forces the normalized rate of $cov=0$ features to exactly 1 (i.e. "no information").

**cytozip implementation notes** (`features.py`):

- With `score='posterior_frac'`, `.X` stores the **un-normalized** posterior mean $\dfrac{mc+\alpha}{cov+\alpha+\beta}$ (still on $[0,1]$, comparable to `frac`).
- Computed **only on covered entries** ($cov>0$); uncovered entries stay implicit 0 (same sparsity as `frac`).
- $\alpha,\beta$ are **per cell** (pooled over that cell's features), written to `adata.obs['alpha','beta','prior_mean','rho']`; degenerate cells (NaN $\alpha/\beta$) fall back to the raw fraction $mc/cov$.

## 9. HVF accumulators, dispersion & normalized dispersion (English)

To select highly variable features (HVF) **without re-reading the matrix** (and even after merging multiple files), three **additive** accumulators are stored per feature $j$. Let $f_{ij}$ be the methylation fraction of cell $i$ at feature $j$ (posterior mean of Section 8 by default, raw $mc/cov$ fallback for degenerate cells; set `hvf_frac='raw'` to use $mc/cov$ directly), summed only over covering cells ($cov_{ij}>0$):

$$
\text{hvf\_n\_cov}_j = \sum_i \mathbb{1}(cov_{ij}>0), \qquad
\text{hvf\_sum}_j = \sum_i f_{ij}, \qquad
\text{hvf\_sum\_sq}_j = \sum_i f_{ij}^2
$$

All three are sums over cells and thus **additive across cells and across datasets** (just add per feature when merging); they are written to `adata.var`. From them:

**Mean and variance**:

$$
\text{mean}_j = \frac{\text{hvf\_sum}_j}{\text{hvf\_n\_cov}_j}, \qquad
\text{var}_j = \frac{\text{hvf\_sum\_sq}_j}{\text{hvf\_n\_cov}_j} - \text{mean}_j^2
$$

**Dispersion**:

$$
\text{disp}_j = \frac{\text{var}_j}{\text{mean}_j}
$$

**Normalized dispersion (scanpy 'seurat' flavor)**: bin all features into $K$ mean-bins (e.g. $K=20$) and z-score the dispersion **within each bin**, avoiding a bias toward high-mean features:

$$
\text{ndisp}_j = \frac{\text{disp}_j - \mu_{b(j)}}{\sigma_{b(j)}}
$$

where $b(j)$ is feature $j$'s mean-bin and $\mu_{b},\sigma_{b}$ are the mean and std of dispersion within that bin. The top-$n$ features by $\text{ndisp}$ are the HVF.

**Why store only the three accumulators**: `mean/var/dispersion` can be reconstructed exactly from $(\text{hvf\_n\_cov},\text{hvf\_sum},\text{hvf\_sum\_sq})$, and those three are additive — so after merging several `.h5ad` files (e.g. `AnnDataCollection.from_files(..., var_agg='sum')`) the HVF can be recomputed correctly on the **combined cells**. **Note**: `mean/var/dispersion` themselves are **not additive** — do not sum them directly; in particular the `normalized dispersion` depends on the binning over all features and must be computed **last**, once. Features covered by fewer than a threshold of cells (e.g. `min_cells`) are marked ineligible during selection.

---

# 细胞类型分类器（`CellTypeClassifier` / `predict_cell_type`，位点似然 / 朴素贝叶斯）

面向**浅测序 snm3C-seq** 单细胞的细胞类型预测（`cytozip/model.py`）。给定一个单细胞 query 的逐位点 `mc`/`cov`（**所有胞嘧啶**，CpG 与 CpH 混在一起）以及若干**细胞类型级 pseudobulk**（深测序参考）`.cz`，判断该 query 细胞最可能属于哪个细胞类型。所有输入 `.cz` 必须对齐到**同一参考轴**（每行一个参考胞嘧啶、顺序一致），因此位点靠行索引在 query 与各参考间对齐。

## 10. 参考频率的 Beta 收缩估计（`estimate_theta`）

对每个判别性位点 $c$ 、候选类型 $t$ ，用第 3 节的矩估计先验 $\text{Beta}(\alpha_0,\beta_0)$ 对参考甲基化频率做**收缩估计**（等价于第 8 节的后验均值）：

$$
\theta_{c,t} = \frac{m_{c,t} + \alpha_0}{n_{c,t} + \alpha_0 + \beta_0}
$$

其中 $m_{c,t}$ 为该类型在位点 $c$ 的甲基化计数、 $n_{c,t}$ 为覆盖。要求 $\alpha_0,\beta_0>0$ ，从而 $\theta\in(0,1)$ 开区间， $\log\theta$ 、 $\log(1-\theta)$ 不发散。频率**保持连续、绝不二值化**——判别信号恰恰藏在中间频率里。

## 11. CpG / CpH 双通道

CpG 与 CpH 的背景甲基化完全不同，故建成**两个独立通道**，各自有自己的频率与收缩先验（ $\alpha_0,\beta_0$ 由 `estimate_beta_prior` 在**该通道池化的参考计数**上分别估计）。CpG/CpH 的划分来自 `context` 列的第 2 个碱基： $G\Rightarrow$ CpG（如 `CGN`）， $A/C/T\Rightarrow$ CpH（如 `CAC`），其它（如 `CNN`）两者都不属于、丢弃。因此**单个装了所有胞嘧啶的 `.cz` 就够了**，无需预先拆成 CG / CH 两个文件；context 通常来自单独的 `build_ref` 参考（`reference=`，含 `pos, strand, context`），因为细胞类型文件一般只存 `mc`/`cov`。

## 12. 判别性位点选择（`_select_discriminative`）

每个位点的判别力用**跨类型频率极差**打分：

$$
s_c = \max_t \theta_{c,t} - \min_t \theta_{c,t}
$$

各类型分歧最大的位点携带最多分类信息，而甲基化恒定（ $s_c\approx 0$ ）的位点无信息、丢弃。`top=None` 时保留所有 $s_c \ge \text{min\_range}$ 的位点；`top` 为整数则取分数最高的 top-$N$ ；`top` 为 $(0,1]$ 的浮点则取最高的那一比例。CpG / CpH 各自独立选择。

## 13. 位点似然打分（每通道）

query 细胞在位点 $c$ 观测到 $mc_c$ 个甲基化、 $cov_c$ 次覆盖（ $umc_c = cov_c - mc_c$ 为未甲基化）。在类型 $t$ 下，把每个位点当作 **Bernoulli** 观测并在选中的判别位点上聚合对数似然：

$$
\ell_t = \sum_c \Bigl[\, mc_c\,\log\theta_{c,t} + (cov_c - mc_c)\,\log(1-\theta_{c,t}) \,\Bigr]
$$

实现上预存 $\log\theta$ 与 $\log(1-\theta)$ 矩阵（ $n_\text{sites}\times n_\text{types}$ ），打分即两次矩阵-向量乘： $\ell = mc \cdot \log\theta + umc \cdot \log(1-\theta)$ 。

## 14. 通道合并、丰度先验与后验

两通道在对数空间加权求和，并加上类型对数先验：

$$
\log \text{post}_t = \log\pi_t + \lambda_\text{cg}\,\ell^{\text{cg}}_t + \lambda_\text{ch}\,\ell^{\text{ch}}_t
$$

- $\lambda_\text{cg},\lambda_\text{ch}$ 为通道权重（默认各 1）。CpH 位点远多于 CpG，容易隐性主导；可调低 $\lambda_\text{ch}$ 再平衡。
- **丰度 / 温度先验**： $\pi_t \propto N_t^{\,\text{prior\_alpha}}$ ，其中 $N_t$ 为该类型的参考细胞数。`prior_alpha=0` → 均匀先验；`prior_alpha=1` → 纯丰度先验；未提供 `cell_counts` 时恒为均匀先验 $\pi_t = 1/T$ 。

对类型做**数值稳定的 softmax** 得到标定后的后验概率：

$$
P(t \mid \text{cell}) = \frac{\exp(\log\text{post}_t - \max_{t'}\log\text{post}_{t'})}{\sum_{t''}\exp(\log\text{post}_{t''} - \max_{t'}\log\text{post}_{t'})}
$$

**预测与弃权**：取 $\hat t = \arg\max_t P(t\mid\text{cell})$ ，置信度为 $\max_t P$ 。若给定 `abstain_threshold` 且最高概率低于它，则标签置为 `'unassigned'`（弃权）。

**接口**：`CellTypeClassifier.fit()` 在 pseudobulk 上估计并选点，`predict` / `predict_proba` / `predict_batch` 打分；`save` / `load` 用单个 `.npz`（不 pickle，仅存 $\log\theta$ 、 $\log(1-\theta)$ 、掩码、判别位点索引与元数据 JSON）。`predict_cell_type(...)` 是"拟合 + 预测一个细胞"的一步式便捷封装。

## 15. Bulk 去卷积（`deconvolve` / `deconvolve_bulk`，参考型细胞比例估计）

同一份参考频率矩阵 $\theta_{c,t}$ （第 10 节）还能反过来用：给定一个 **bulk** 样本（bulk WGBS 或甲基化芯片）在各位点的甲基化水平，估计它由各细胞类型按什么**比例** $f_t$ 混合而成——即参考型甲基化去卷积（Houseman / CIBERSORT 一类）。

**混合模型**：bulk 在位点 $c$ 的甲基化水平是各类型频率按比例的线性混合：

$$
\beta_c \approx \sum_t f_t\, \theta_{c,t}, \qquad f_t \ge 0,\ \sum_t f_t = 1
$$

其中 WGBS 的 $\beta_c = mc_c / cov_c$ ，芯片则直接是 β 值。

**求解（约束加权最小二乘）**：以 bulk 覆盖为权重 $w_c$ （芯片无覆盖时取 $w_c=1$ ，退化为普通最小二乘），令 $A = \sqrt{w}\odot\Theta$ 、 $b = \sqrt{w}\odot\beta$ ，求解

$$
\hat f = \arg\min_{f}\ \lVert A f - b \rVert^2 \quad \text{s.t.}\quad f_t \ge 0,\ \textstyle\sum_t f_t = 1
$$

- **非负性**由 NNLS 保证；**和为 1**（`sum_to_one=True`，默认）由 SLSQP 施加等式约束，得到一个完整单纯形上的解。深覆盖位点权重更大（ $w_c = cov_c$ ），因此比未加权更稳健。
- **未知成分**（`allow_unknown=True`）：把等式放宽为 $\sum_t f_t \le 1$ ，余量 $1 - \sum_t f_t$ 作为参考里没有的"unknown"细胞类型单独报告——适用于参考不完整的样本。
- `sum_to_one=False` 则退化为纯 NNLS（比例不必和为 1），仅作诊断用。

**位点选择即 marker 选择**：去卷积只用第 12 节选出的判别性位点（跨类型极差 $s_c$ 大者），这正是细胞类型 marker；`contexts` 可选只用 `'cg'`（默认，兼容芯片；WGBS 惯例亦然）、`'ch'` 或 `'cg+ch'` 两通道联合。仅使用 bulk 实际覆盖（ $cov_c \ge$ `min_cov`）的位点。

**拟合优度**：报告（加权）判定系数

$$
R^2 = 1 - \frac{\sum_c w_c\,(\hat\beta_c - \beta_c)^2}{\sum_c w_c\,(\beta_c - \bar\beta)^2},\qquad \hat\beta_c = \sum_t \hat f_t\,\theta_{c,t}
$$

用于判断混合模型对该 bulk 的解释力。

**接口**：`CellTypeClassifier.deconvolve()`（单个 bulk → 比例 `Series`）、`deconvolve_batch()`（多个 bulk → 比例 DataFrame）、`deconvolve_multicell()`（一个 cat `.cz` 内打包的多个 bulk）；`deconvolve_bulk(...)` 是"拟合参考 + 去卷积"的一步式便捷封装，与 `predict_cell_type(...)` 用法对齐。

## 16. 用未甲基化计数模拟 ATAC 做 peak calling（`call_peaks` / `call_peaks_bdg`）

**生物学原理**：CpG 低甲基化区（hypomethylation / mCG valley）标记开放染色质与调控元件（启动子/增强子），是 snmC-seq 推断开放染色质的公认原理。于是把每个位点的**未甲基化计数** $umc_c = cov_c - mc_c$ 当作类 ATAC 的"reads 数"信号，用 MACS3 找 peak。

**关键：覆盖对照**。 $umc = cov - mc$ 与测序深度、CpG 密度强共线，直接找 peak 会把深覆盖 / CpG island 区误判为 peak。因此把**总覆盖 $cov$ 作为 control（input）轨**，让 peak 反映 umc 相对期望的**局部富集**（即局部 unmeth 比例高于全局），而非原始深度。全局未甲基化率

$$
r = \frac{\sum_c umc_c}{\sum_c cov_c}
$$

给出每个位点的**期望未甲基化 pileup** $\;\widehat{umc}(x) = r\cdot cov_\text{pileup}(x)$ （均匀甲基化零假设下）。

**两条实现路线**：

1. **伪 read 路线 `call_peaks`**：每个位点按 $umc_c$ 展开成 $umc_c$ 条长 `fragment_size`、以位点为中心的伪 reads（BED），`control='cov'` 时同样把 $cov_c$ 展开成 control BED；喂给 `macs3 callpeak --nomodel --extsize fragment_size -q <qvalue> [-c control]`。直观、可直接出 `.narrowPeak`，但伪 reads 总数 $=\sum_c umc_c$ ，深 pseudobulk 会很大。

2. **bedGraph 路线 `call_peaks_bdg`（省内存，推荐深 pseudobulk）**：用**差分数组**把每个位点的计数摊到 $[x-\text{ext}/2,\ x+\text{ext}/2)$ 并求和，直接得到分段常值 pileup（内存 $O(n_\text{sites})$ ，与 $\sum umc$ 无关）。treatment = $umc$ pileup，control(lambda) = $cov$ pileup $\times r$ ；再依次

$$
\texttt{bdgopt (}\times r\texttt{)} \;\to\; \texttt{bdgcmp -m ppois} \;\to\; \texttt{bdgpeakcall}
$$

其中 `ppois` 对每个位点给出观测 pileup 相对期望 lambda 的 **Poisson 检验** $-\log_{10} p$ 分数，`bdgpeakcall -c <cutoff> -l <min\_len> -g <max\_gap>` 阈值化并合并成 peak。

**注意点**：只用 **CpG**（`index=` 限定；CpH 是另一套信号）；必须在 **pseudobulk**（合并同类型细胞的单轨 `.cz`）上做，单细胞太稀疏（两函数都强制单轨）；MACS3 的 Poisson 背景模型对 $umc\sim\text{Binomial}(cov,1-p)$ 只是近似，q/p 值作排序阈值用、不宜当精确 FDR。

**接口**：`call_peaks(..., control='cov')`（伪 read + 覆盖对照）与 `call_peaks_bdg(..., control='cov')`（bedGraph + `bdgcmp`/`bdgpeakcall`），均支持 `signal='unmeth'|'meth'`、`index=`（context 过滤）、`min_cov`。

---

# Cell-type classifier (`CellTypeClassifier` / `predict_cell_type`, site-likelihood / naive Bayes) — English

Cell-type prediction for **shallow snm3C-seq** single cells (`cytozip/model.py`). Given a query cell's per-cytosine `mc`/`cov` (**all cytosines**, CpG and CpH together) and a set of **cell-type pseudobulk** (deep reference) `.cz` files, predict which cell type the query most likely belongs to. All inputs must be aligned to the **same reference axis** (one row per reference cytosine, in order), so positions align by row index across the query and references.

## 10. Beta-shrinkage estimate of the reference frequency (`estimate_theta`)

For each discriminative site $c$ and candidate type $t$, estimate the reference methylation frequency with the Section-3 MoM prior $\text{Beta}(\alpha_0,\beta_0)$ by shrinkage (equivalent to the Section-8 posterior mean):

$$
\theta_{c,t} = \frac{m_{c,t} + \alpha_0}{n_{c,t} + \alpha_0 + \beta_0}
$$

with $m_{c,t}$ the methylated count and $n_{c,t}$ the coverage of that type at site $c$. Both $\alpha_0,\beta_0>0$ keep $\theta\in(0,1)$ so $\log\theta$ and $\log(1-\theta)$ never diverge. The frequency is kept **continuous, never binarized** — the discriminative signal lives in the intermediate frequencies.

## 11. CpG / CpH two channels

CpG and CpH have very different backgrounds, so they are modeled as **two independent channels**, each with its own frequencies and shrinkage prior ($\alpha_0,\beta_0$ estimated per context by `estimate_beta_prior` on that context's pooled reference counts). The CpG/CpH split comes from the 2nd base of the `context` column: $G\Rightarrow$ CpG (e.g. `CGN`), $A/C/T\Rightarrow$ CpH (e.g. `CAC`), anything else (e.g. `CNN`) belongs to neither and is dropped. Hence a **single `.cz` carrying all cytosines is enough** — no need to pre-split CG / CH; context usually comes from a separate `build_ref` reference (`reference=`, with `pos, strand, context`) because the cell-type files typically store only `mc`/`cov`.

## 12. Discriminative-site selection (`_select_discriminative`)

Each site's discriminative power is scored by the **across-type frequency range**:

$$
s_c = \max_t \theta_{c,t} - \min_t \theta_{c,t}
$$

Sites where types disagree most carry the most classification signal, while constant-methylation sites ($s_c\approx 0$) are uninformative and dropped. `top=None` keeps all sites with $s_c \ge \text{min\_range}$; an integer `top` keeps the top-$N$ by score; a float in $(0,1]$ keeps the top fraction. CpG and CpH are selected independently.

## 13. Site-likelihood scoring (per channel)

The query cell observes $mc_c$ methylated calls out of $cov_c$ at site $c$ ($umc_c = cov_c - mc_c$ unmethylated). Under type $t$, treat each site as a **Bernoulli** observation and aggregate the log-likelihood over the selected discriminative sites:

$$
\ell_t = \sum_c \Bigl[\, mc_c\,\log\theta_{c,t} + (cov_c - mc_c)\,\log(1-\theta_{c,t}) \,\Bigr]
$$

In practice the $\log\theta$ and $\log(1-\theta)$ matrices ($n_\text{sites}\times n_\text{types}$) are precomputed, so scoring is two matrix-vector products: $\ell = mc \cdot \log\theta + umc \cdot \log(1-\theta)$.

## 14. Channel combination, abundance prior & posterior

The two channels are summed in log-space with weights, plus a type log-prior:

$$
\log \text{post}_t = \log\pi_t + \lambda_\text{cg}\,\ell^{\text{cg}}_t + \lambda_\text{ch}\,\ell^{\text{ch}}_t
$$

- $\lambda_\text{cg},\lambda_\text{ch}$ are channel weights (default 1 each). CpH sites vastly outnumber CpG and can implicitly dominate; lower $\lambda_\text{ch}$ to rebalance.
- **Abundance / temperature prior**: $\pi_t \propto N_t^{\,\text{prior\_alpha}}$, with $N_t$ the reference cell count of type $t$. `prior_alpha=0` → uniform prior; `prior_alpha=1` → pure abundance; without `cell_counts` the prior is always uniform $\pi_t = 1/T$.

A **numerically stable softmax** over types yields calibrated posterior probabilities:

$$
P(t \mid \text{cell}) = \frac{\exp(\log\text{post}_t - \max_{t'}\log\text{post}_{t'})}{\sum_{t''}\exp(\log\text{post}_{t''} - \max_{t'}\log\text{post}_{t'})}
$$

**Prediction and abstention**: take $\hat t = \arg\max_t P(t\mid\text{cell})$ with confidence $\max_t P$. If `abstain_threshold` is given and the top probability is below it, the label becomes `'unassigned'` (abstention).

**API**: `CellTypeClassifier.fit()` estimates and selects sites on the pseudobulks; `predict` / `predict_proba` / `predict_batch` score cells; `save` / `load` use a single `.npz` (no pickling — only $\log\theta$, $\log(1-\theta)$, masks, discriminative-site indices, and a JSON metadata string). `predict_cell_type(...)` is a one-shot "fit + predict one cell" convenience wrapper.

## 15. Bulk deconvolution (`deconvolve` / `deconvolve_bulk`, reference-based fraction estimation)

The same reference frequency matrix $\theta_{c,t}$ (Section 10) can be used the other way round: given a **bulk** sample (bulk WGBS or a methylation array) with a methylation level at each site, estimate the **fractions** $f_t$ of each cell type it is a mixture of — i.e. reference-based methylation deconvolution (Houseman / CIBERSORT style).

**Mixture model**: the bulk level at site $c$ is a fraction-weighted linear mixture of the per-type frequencies:

$$
\beta_c \approx \sum_t f_t\, \theta_{c,t}, \qquad f_t \ge 0,\ \sum_t f_t = 1
$$

with $\beta_c = mc_c / cov_c$ for WGBS, or the array beta value directly.

**Solve (constrained weighted least squares)**: weighting each site by the bulk coverage $w_c$ (or $w_c=1$ for arrays, recovering ordinary least squares), let $A = \sqrt{w}\odot\Theta$ and $b = \sqrt{w}\odot\beta$ and solve

$$
\hat f = \arg\min_{f}\ \lVert A f - b \rVert^2 \quad \text{s.t.}\quad f_t \ge 0,\ \textstyle\sum_t f_t = 1
$$

- **Non-negativity** is enforced by NNLS; **sum-to-one** (`sum_to_one=True`, default) is imposed by SLSQP as an equality constraint, giving a solution on the full simplex. Deeply covered sites carry more weight ($w_c = cov_c$), so the fit is more robust than unweighted.
- **Unknown compartment** (`allow_unknown=True`): relax the equality to $\sum_t f_t \le 1$ and report the remainder $1 - \sum_t f_t$ as an ``unknown`` cell type absent from the reference — useful when the reference is incomplete.
- `sum_to_one=False` reduces to plain NNLS (fractions need not sum to 1), for diagnostics only.

**Site selection is marker selection**: deconvolution uses only the discriminative sites from Section 12 (large across-type range $s_c$), which are exactly the cell-type markers; `contexts` chooses `'cg'` (default, the only option for arrays and the usual one for WGBS), `'ch'`, or `'cg+ch'` (both channels jointly). Only sites the bulk actually covers ($cov_c \ge$ `min_cov`) are used.

**Goodness of fit**: the (weighted) coefficient of determination

$$
R^2 = 1 - \frac{\sum_c w_c\,(\hat\beta_c - \beta_c)^2}{\sum_c w_c\,(\beta_c - \bar\beta)^2},\qquad \hat\beta_c = \sum_t \hat f_t\,\theta_{c,t}
$$

reports how well the mixture explains the bulk.

**API**: `CellTypeClassifier.deconvolve()` (one bulk → a fraction `Series`), `deconvolve_batch()` (many bulks → a fraction DataFrame), `deconvolve_multicell()` (many bulks packed in one cat `.cz`); `deconvolve_bulk(...)` is a one-shot "fit reference + deconvolve" convenience wrapper mirroring `predict_cell_type(...)`.

## 16. Peak calling from unmethylated counts, ATAC-style (`call_peaks` / `call_peaks_bdg`)

**Biological rationale**: CpG hypomethylation (mCG valleys) marks open chromatin and regulatory elements (promoters/enhancers) — the standard way to infer open chromatin from snmC-seq. So the per-site **unmethylated count** $umc_c = cov_c - mc_c$ is used as an ATAC-like "read count" signal and fed to MACS3 for peak calling.

**Key: a coverage control**. $umc = cov - mc$ is strongly confounded by sequencing depth and cytosine density, so calling peaks on it directly flags deeply covered / CpG-island regions. Passing the total coverage $cov$ as the **control (input) track** makes peaks reflect a genuine **local enrichment** of unmethylation (local unmeth fraction above the global rate) rather than raw depth. The global unmethylation rate

$$
r = \frac{\sum_c umc_c}{\sum_c cov_c}
$$

gives the **expected unmethylated pileup** $\;\widehat{umc}(x) = r\cdot cov_\text{pileup}(x)$ under a uniform-methylation null.

**Two implementations**:

1. **Pseudo-read route `call_peaks`**: each site is expanded into $umc_c$ reads of length `fragment_size` centred on it (BED); with `control='cov'` the coverage $cov_c$ is expanded the same way as a control BED; both go to `macs3 callpeak --nomodel --extsize fragment_size -q <qvalue> [-c control]`. Intuitive and emits `.narrowPeak` directly, but the read count is $\sum_c umc_c$, which explodes on deep pseudobulks.

2. **bedGraph route `call_peaks_bdg` (memory-efficient, preferred for deep pseudobulks)**: a **difference array** spreads each site's count over $[x-\text{ext}/2,\ x+\text{ext}/2)$ and sums, yielding a piecewise-constant pileup in $O(n_\text{sites})$ memory (independent of $\sum umc$). Treatment = $umc$ pileup, control (lambda) = $cov$ pileup $\times r$; then

$$
\texttt{bdgopt (}\times r\texttt{)} \;\to\; \texttt{bdgcmp -m ppois} \;\to\; \texttt{bdgpeakcall}
$$

where `ppois` scores each position by the **Poisson p-value** ($-\log_{10} p$) of the observed pileup against the expected lambda, and `bdgpeakcall -c <cutoff> -l <min_len> -g <max_gap>` thresholds and merges into peaks.

**Caveats**: use **CpG only** (`index=`; CpH is a different signal); run on a **pseudobulk** (a single-track `.cz` of merged same-type cells) — single cells are too sparse (both functions enforce a single track); MACS3's Poisson background is only an approximation to $umc\sim\text{Binomial}(cov, 1-p)$, so treat p/q as ranking thresholds, not exact FDR.

**API**: `call_peaks(..., control='cov')` (pseudo-reads + coverage control) and `call_peaks_bdg(..., control='cov')` (bedGraph + `bdgcmp`/`bdgpeakcall`); both take `signal='unmeth'|'meth'`, `index=` (context filter), and `min_cov`.


