# Beta-Binomial 矩估计（估计甲基化 Beta 先验的 α, β）

## 1. 模型与参数化

每个位点 $i$ 有覆盖 $n_i$、甲基化计数 $k_i$。假设位点真实甲基化率 $p_i \sim \text{Beta}(\alpha, \beta)$，观测 $k_i \mid p_i \sim \text{Binomial}(n_i, p_i)$。

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

比纯 Binomial 多了过离散因子 $[1 + (n_i - 1)\rho]$——这正是"用 cov 校正"的来源。$\rho = 0$ 退化为 Binomial。

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

所以 $E[X] = N + \rho \sum_i (n_i - 1)$（$N$ = 位点数）。令 $X = E[X]$ 解出

$$
\boxed{\ \hat\rho = \frac{X - N}{\sum_i (n_i - 1)}\ }
$$

（更严的自由度修正把分子 $N$ 换成 $N-1$，即 Tarone 1979 的形式；数据量大时差别可忽略。）

**转成 α, β**：

$$
\hat\kappa = \frac{1 - \hat\rho}{\hat\rho}, \qquad
\hat\alpha = \hat\mu\, \hat\kappa, \qquad
\hat\beta = (1 - \hat\mu)\, \hat\kappa
$$

## 4. 等价的"频率方差"直觉版（覆盖近似相等时）

设 $f_i = k_i / n_i$，则

$$
\operatorname{Var}(f_i) = \mu(1-\mu)\left[\frac{1}{n_i} + \left(1 - \frac{1}{n_i}\right)\rho\right]
$$

若各位点覆盖 $\approx \bar n$，用观测方差 $s^2 = \operatorname{Var}(f_i)$ 解：

$$
\hat\rho = \frac{\dfrac{s^2}{\mu(1-\mu)} - \dfrac{1}{\bar n}}{1 - \dfrac{1}{\bar n}}
$$

这就明确显示了"从率方差里**扣掉 binomial 抽样项 $1/\bar n$**"再得到真实的 Beta 离散——这是它比 ALLCools 纯 Beta-MoM（不扣抽样噪声）更准的地方。覆盖差异大时应该用第 3 节的 $X$ 版本（对每个 $n_i$ 加权），不要用这个近似。

## 5. 边界与实现要点

- **裁剪** $\hat\rho \in (\epsilon,\ 1-\epsilon)$：
  - $\hat\rho \le 0$（欠离散 / 被抽样噪声压过）→ 令 $\rho \to \epsilon$，即 $\kappa$ 很大（强 prior）；
  - $\hat\rho \to 1$ → $\kappa \to 0$（几乎无收缩）。
- $\hat\mu$ 也裁剪到 $(\epsilon, 1-\epsilon)$，避免 $\log$ 发散。
- **过滤 $n_i = 0$**，最好 $n_i \ge 2$（$n_i = 1$ 对 $\sum(n_i - 1)$ 无贡献且噪声大）。
- **CpG / CpH 分开各估一套**（背景完全不同）。
- 在**深测序 pseudobulk 参考**上估，而非浅测序 query cell。

## 附：与 ALLCools 的区别

ALLCools（`calculate_posterior_mc_frac`）用的是**纯 Beta 矩估计**：直接对观测率 $f_i = mc/cov$ 求均值 $\mu$ 和方差 $\sigma^2$，套用

$$
\alpha = \mu\left[\frac{\mu(1-\mu)}{\sigma^2} - 1\right], \qquad
\beta = \alpha\left(\frac{1}{\mu} - 1\right)
$$

它**没有扣除 binomial 抽样噪声**（$\sigma^2$ 里混了抽样方差），所以在低覆盖数据上会高估方差、低估浓度 $\kappa$。上面的 Beta-Binomial 版本通过第 2 节的过离散因子对覆盖做了校正。

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

覆盖越低（$n_i$ 越小），抽样噪声项越大，旧方法的偏差越严重——而单细胞甲基化恰恰是低覆盖场景。

### 6.2 正确处理不等覆盖（按 $n_i$ 加权）

- 旧方法对每个位点的 $f_i$ **等权**求均值/方差：一个 $cov=1$ 的位点（$f_i$ 只能取 0 或 1，噪声极大）和一个 $cov=100$ 的位点被同等对待。
- 新方法的 $\hat\mu=\sum k_i/\sum n_i$ 是**按覆盖加权**的池化估计，$X$ 统计量对每个 $n_i$ 单独加权，天然地让高覆盖位点贡献更多、低覆盖位点贡献更少。

### 6.3 均值估计更稳、无 $0/0$ 病态

- 旧方法先对每个位点算 $f_i=mc/cov$ 再平均，$cov=0$ 会产生 `NaN`、$cov$ 很小时 $f_i$ 抖动剧烈。
- 新方法先求和再相除（$\sum k_i / \sum n_i$），对 $cov=0$ 的位点自然免疫，估计更平滑。

### 6.4 参数有明确、可解释的物理含义

新方法把先验拆成 $\mu$（背景甲基化水平）和 $\rho$（位点内相关 / 过离散），二者都可解释、可分别裁剪；CpG / CpH 背景差异也能自然体现在各自的 $\rho$ 上。

### 6.5 下游后验一致

先验 $\alpha,\beta$ 更准 $\Rightarrow$ 共轭后验 $\text{Beta}(\alpha+k_i,\ \beta+n_i-k_i)$ 的**收缩强度更合理**：低覆盖位点向背景 $\mu$ 收缩得当，高覆盖位点保留自身信息。后验均值 `post_frac` 和后验标准差 `post_sigma` 的公式不变，但因为 $\alpha,\beta$ 更准，数值也更可靠。

### 6.6 什么时候两者差别不大

当所有位点覆盖都很高且大致相等（$n_i \gg 1$、$\bar n$ 接近）时，抽样噪声项 $1/n_i \to 0$，两种方法趋于一致。**差别主要出现在低覆盖、覆盖高度不均的数据上**——也就是单细胞甲基化的典型情形，因此推荐新方法。

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

有了每个细胞的 Beta 先验 $\text{Beta}(\alpha,\beta)$（第 3 节的矩估计）后，某位点/特征观测到 $mc$ 个甲基化、覆盖 $cov$，由 Beta–Binomial 的**共轭性**得到该位点真实甲基化率的后验：

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

**可选的按细胞归一化（ALLCools 口径）**：$\widehat{p}_{\text{post}} / \mu$，使 $cov=0$ 的特征归一化率恒为 1（即"无信息"）。

**cytozip 实现要点**（`features.py`）：

- `score='posterior_frac'` 时 `.X` 存**未归一化**的后验均值 $\dfrac{mc+\alpha}{cov+\alpha+\beta}$（尺度仍在 $[0,1]$，与 `frac` 可比）。
- 仅在**覆盖到的条目**（$cov>0$）上计算；未覆盖条目保持隐式 0（与 `frac` 相同的稀疏结构）。
- $\alpha,\beta$ 为**每个细胞一套**（对该细胞的所有特征池化），写入 `adata.obs['alpha','beta','prior_mean','rho']`；退化细胞（$\alpha/\beta$ 为 NaN）回退到原始分数 $mc/cov$。

## 9. 高变特征累加量与离散度（HVF：`hvf_n_cov` / `hvf_sum` / `hvf_sum_sq`、var、dispersion、normalized dispersion）

为了在**不重新读取矩阵**（甚至跨多个文件合并）的前提下选出高变特征（HVF），对每个特征 $j$ 存三个**可加**的累加量。设 $f_{ij}$ 为细胞 $i$ 在特征 $j$ 上的甲基化分数（默认用第 8 节的后验均值，退化细胞回退原始 $mc/cov$；也可设 `hvf_frac='raw'` 直接用 $mc/cov$），只在覆盖到的细胞（$cov_{ij}>0$）上累加：

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

**归一化离散度（normalized dispersion，scanpy 'seurat' 口径）**：把所有特征按 $\text{mean}$ 分到 $K$ 个箱（例如 $K=20$），在**每个箱内**对 dispersion 做 z-score，避免只挑到高均值特征：

$$
\text{ndisp}_j = \frac{\text{disp}_j - \mu_{b(j)}}{\sigma_{b(j)}}
$$

其中 $b(j)$ 是特征 $j$ 所在的均值箱，$\mu_{b},\sigma_{b}$ 是该箱内 dispersion 的均值与标准差。最后取 $\text{ndisp}$ 最高的 top-$n$ 个特征作为 HVF。

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


