# Value-at-Risk (VaR) and Conditional VaR (CVaR)

## Introduction

**Value-at-Risk (VaR)** is a widely used risk measure that answers the question:  
*"What is the maximum expected loss of a portfolio over a given horizon, at a certain confidence level?"* 

In simple terms:

`A daily VaR at 95% of 2.81% means there is a 95% chance that your portfolio will not lose more than 2.81% during the next day.`

Conversely, there is a 5% chance that your portfolio will lose more than 2.81%.

Formally, the $\alpha$-VaR is the quantile of the loss distribution:  

$$
P(L > \text{VaR}_\alpha) = 1-\alpha
$$

But VaR only gives a threshold loss and does not provide information on the magnitude of losses beyond this threshold.

**Conditional VaR (CVaR)**, also called *Expected Shortfall (ES)*, addresses this limitation. It measures the average loss given that losses exceed the VaR:

$$
\text{CVaR}_\alpha = \mathbb{E}[L \mid L \geq \text{VaR}_\alpha].
$$

Unlike VaR, CVaR is a coherent risk measure, meaning it respects properties such as sub-additivity, making it more robust for portfolio risk management.

---

## Methods for VaR Estimation

### 1. Historical Simulation VaR
- Uses empirical distribution of past returns.  
- VaR is directly the empirical quantile of historical losses.  

Advantage: Non-parametric, captures fat tails and asymmetry.  
Drawback: Heavily dependent on past data; assumes the past reflects the future.  

---

### 2. Parametric (Variance-Covariance) VaR
- Assumes returns are normally distributed with mean $\mu$ and volatility $\sigma$.  
- VaR is computed as:  

$$
\text{VaR}_{\alpha} = -(\mu + z_{\alpha} \times \sigma)
$$

where $z_\alpha$ is the quantile of the standard normal distribution.  

Advantage: Very fast and easy to compute.  
Drawback: Assumes normality, underestimates tail risk in case of skewness or fat tails.  

---

### 3. Cornish-Fisher VaR
- Extension of parametric VaR that adjusts quantiles to account for skewness and kurtosis in returns.  
- Modified quantile:  

$$
z_\alpha^{CF} = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3)K - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2
$$

where $S$ is skewness and $K$ is excess kurtosis (kurtosis - 3).  
- Once the adjusted quantile is calculated, VaR is calculated as in the parametric approach

Advantage: Captures non-normality (skewed/fat-tailed returns).  
Drawback: Still an approximation; accuracy decreases for extreme tails.  

---

### 4. EWMA (Exponentially Weighted Moving Average) VaR
- Assumes conditional volatility evolves over time with more weight on recent returns.  
- Volatility is estimated recursively with a decay factor $\lambda$:

$$
\sigma^2_t = \lambda \times \sigma^2_{t-1} + (1-\lambda) \times r^2_{t-1}
$$
 
Where:
- $\sigma^2_t$ is the estimated variance for day $t$
- $r^2_{t-1}$ is the squared return of the previous day
- $\lambda$ is the smoothing factor (decay factor), typically between 0.9 and 0.98
 
The value $\lambda = 0.94$ is often used (recommended by RiskMetrics).
A lambda of 0.94 in EWMA means the model keeps 94% of past volatility and gives 6% weight to new returns.
The half-life of ~11 days ($\frac{\ln(0.5)}{\ln(0.94)}$) means it takes about 11 days for a shock’s impact to reduce by half.
This gives moderate reactivity: the model reacts to market shocks but smooths out short-term noise.

Once EWMA volatility is calculated, VaR is calculated as in the parametric approach

Advantage: Adapts to volatility clustering, more responsive to shocks.  
Drawback: Still assumes (conditional) normality; choice of $\lambda$ affects results.  

---

## Conditional VaR (Expected Shortfall)
  
- CVaR answers *"If things go worse than VaR, what is my average loss?"*. 
- CVaR is computed as:

$$
\text{CVaR}_\alpha = \mu + \sigma \frac{\phi(z_\alpha)}{\alpha}
$$
$$

where $\phi(\cdot)$ is the standard normal PDF and $\Phi(\cdot)$ the CDF.  

Advantages: 
- Coherent risk measure (sub-additive, convex).  
- Captures the tail behavior of losses.  

Drawbacks:
- More computationally demanding (requires integration or simulation).  
- Sensitive to data quality, especially in the extreme tail.  
