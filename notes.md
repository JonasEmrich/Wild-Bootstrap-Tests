# Motivation
- often data does not fulfill the iid assumption
- here we focus on not identically distributed data
  - as it occurs in signal and image analysis

# Signal Model

$Y_i = m_i + e_i$

- where $e_i$ are iid zero mean white noise with finite variance $\sigma^2$
- we assume that $m_i$ changes more slowly than the residuals $e_i$

$Y_i = m(x_i) + e_i, \qquad i = 1, \dots, N$
- where $m(x)$ is a reasonably smooth function on $[0, 1]$
- $x_i = \frac{i}{N}$ is the rescaled time

Clearly, $Y_1 \dots Y_N$ have no common distribution, i.e. are not identically distributed, since their means $m_i \dots m_N$ differ
  - resampling the residuals $e_i$ instead


# Bootstrap
1. Collect data $Y_1 \dots Y_N    \qquad  i = 1,\dots, N$
2. Estimate $\hat{m}_i^0    \qquad  i = 1,\dots, N$
3. Compute residuals $\hat{e}_i \qquad  i = 1,\dots, N$
4. Center residuals so that they follow a zero mean distribution $\hat{e}_i^0 = \hat{e}_i - \frac{1}{N} \sum_j{\hat{e}_j}$
5. Generate Bootstrap residuals $e_i^*$ by resampling
6. Compute Boostrap data $\hat{Y}_i^* = \~{m}_i + e_i^*$
   - where $\~{m}_i = \~{m}(x_i)$ is another approximation of the signal

## Wild Bootstrap
- residuals may not be identically distributed
  - may depend on observations $Y_i$

Wild bootstrap: generate bootstrap residuals $e_i^*$ based on **one** residual $\hat{e}_i$
 - generate $e_i^*$ as a random variable (RV) satisfying the first three moments as $\hat{e}_i$
   - using binary variables for this
