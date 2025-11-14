# Chapter 291: TD3 Portfolio Trading

## Introduction to Twin Delayed Deep Deterministic Policy Gradient (TD3)

Deep reinforcement learning has made significant strides in solving continuous control problems, and portfolio management is one of the most natural applications. Among the family of actor-critic algorithms, **Deep Deterministic Policy Gradient (DDPG)** was one of the first methods to handle continuous action spaces effectively. However, DDPG suffers from a well-known pathology: **overestimation bias** in the Q-function, which leads to suboptimal and often unstable policies.

**Twin Delayed Deep Deterministic Policy Gradient (TD3)**, introduced by Fujimoto, Hoof, and Meger in 2018, addresses the core weaknesses of DDPG through three critical modifications:

1. **Clipped Double-Q Learning** (Twin Critics) — maintain two independent critic networks and take the minimum of their Q-value estimates to combat overestimation.
2. **Delayed Policy Updates** — update the actor (policy) network less frequently than the critic networks, allowing the critic to stabilize before guiding the actor.
3. **Target Policy Smoothing** — add clipped noise to the target action used in the Bellman backup, regularizing the value function and preventing the policy from exploiting narrow peaks in the Q-function landscape.

In the context of portfolio trading, TD3 outputs **continuous portfolio weights** that determine how much capital to allocate to each asset at every time step. This chapter explores the mathematical foundations, explains why TD3 is superior to DDPG for volatile cryptocurrency markets, and provides a complete Rust implementation with Bybit data integration.

---

## Mathematical Foundations

### Actor-Critic Framework Recap

In actor-critic methods, the **actor** $\mu_\theta(s)$ is a deterministic policy that maps states $s$ to actions $a$, and the **critic** $Q_\phi(s, a)$ estimates the expected cumulative reward for taking action $a$ in state $s$ and following the policy thereafter.

The objective for the actor is to maximize the expected Q-value:

$$J(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ Q_\phi(s, \mu_\theta(s)) \right]$$

The critic is trained to minimize the Bellman error:

$$\mathcal{L}(\phi) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q_\phi(s, a) - y \right)^2 \right]$$

where $y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))$ is the target, and $\phi'$, $\theta'$ are the parameters of the target networks.

### Twin Critics: Clipped Double-Q Learning

The key insight of TD3 is that function approximation errors in the critic tend to cause **overestimation** of Q-values. This is analogous to the maximization bias in Q-learning. To mitigate this, TD3 maintains two independent critic networks, $Q_{\phi_1}$ and $Q_{\phi_2}$, and computes the target using the minimum:

$$y = r + \gamma \min_{i=1,2} Q_{\phi'_i}(s', \tilde{a}')$$

By taking the minimum of two independently trained estimates, we obtain a **lower bound** on the true Q-value, which counteracts the overestimation tendency. Each critic is trained independently with its own loss:

$$\mathcal{L}(\phi_i) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q_{\phi_i}(s, a) - y \right)^2 \right], \quad i = 1, 2$$

### Target Policy Smoothing

Instead of using the raw target policy action $\mu_{\theta'}(s')$, TD3 adds clipped Gaussian noise:

$$\tilde{a}' = \mu_{\theta'}(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

where $\sigma$ is the smoothing noise standard deviation and $c$ is the clipping bound. This has the effect of **smoothing the Q-function** over similar actions, preventing the policy from exploiting narrow, unreliable peaks in the learned value landscape. The smoothed action is then clipped to the valid action range (e.g., portfolio weights in $[0, 1]$).

### Delayed Policy Updates

In DDPG, the actor and critic are updated at every step. TD3 introduces a **delay**: the actor is updated only once every $d$ critic updates (typically $d = 2$). The target networks are also updated on this same delayed schedule:

$$\phi'_i \leftarrow \tau \phi_i + (1 - \tau) \phi'_i, \quad \theta' \leftarrow \tau \theta + (1 - \tau) \theta'$$

The rationale is that the critic needs time to converge before the actor should change its behavior based on the critic's guidance. Updating the actor too frequently based on an inaccurate critic leads to divergence.

### Complete TD3 Update Algorithm

```
For each training step t:
    Sample batch (s, a, r, s') from replay buffer D

    # Target action with smoothing
    a_tilde = mu_theta'(s') + clip(N(0, sigma), -c, c)
    a_tilde = clip(a_tilde, a_low, a_high)

    # Clipped double-Q target
    y = r + gamma * min(Q_phi1'(s', a_tilde), Q_phi2'(s', a_tilde))

    # Update both critics
    Update phi1 by minimizing (Q_phi1(s, a) - y)^2
    Update phi2 by minimizing (Q_phi2(s, a) - y)^2

    # Delayed policy update
    if t mod d == 0:
        Update theta by maximizing Q_phi1(s, mu_theta(s))
        Soft update target networks
```

---

## Why TD3 Fixes DDPG's Problems for Volatile Crypto Markets

### The Overestimation Problem in Financial Markets

Cryptocurrency markets are notoriously volatile. Prices can swing 10-20% in a single day, creating a noisy reward signal. When DDPG's single critic overestimates Q-values, the actor learns to take overly aggressive positions, believing they are more profitable than they actually are. This is catastrophic in crypto trading because:

1. **False confidence in risky positions**: The overestimated Q-values make the agent think high-leverage or concentrated positions are safer than they are.
2. **Chasing phantom alpha**: The actor optimizes against an inflated value landscape, leading to strategies that look good in training but fail in live markets.
3. **Training instability**: In volatile markets, the single critic's errors compound, leading to oscillating policies that never converge.

### How TD3's Three Mechanisms Help

**Twin Critics** provide a conservative estimate of action values. In the context of a portfolio with BTC and ETH, if one critic thinks going 80% BTC is worth $Q = 5.0$ and the other thinks it is worth $Q = 3.2$, TD3 uses $3.2$. This prevents the agent from overcommitting to positions based on optimistic errors.

**Delayed Policy Updates** are particularly valuable in financial markets where the signal-to-noise ratio is low. By giving the critic more updates to digest market data before changing the portfolio allocation strategy, TD3 produces smoother, more deliberate rebalancing decisions. This also reduces transaction costs, since the policy changes less frequently.

**Target Policy Smoothing** prevents the agent from exploiting spurious peaks in the Q-landscape. In financial markets, there may be narrow regions of the weight space where the Q-function is artificially high due to fitting noise. By smoothing the target, TD3 favors robust allocations that work well across a range of nearby weight configurations.

### Empirical Observations

In practice, TD3 agents for crypto portfolio trading tend to:
- Produce more stable portfolio weight trajectories
- Achieve lower maximum drawdown compared to DDPG
- Show Q-value estimates that more closely match realized returns
- Require fewer hyperparameter tuning iterations to find a working configuration

---

## Applications: Continuous Portfolio Rebalancing

### Problem Formulation

We model portfolio rebalancing as a Markov Decision Process (MDP):

- **State** $s_t$: A feature vector containing recent price returns, rolling volatilities, volume data, and current portfolio weights for $N$ assets.
- **Action** $a_t$: A vector of target portfolio weights $w \in \mathbb{R}^N$ where $w_i \in [0, 1]$ and $\sum_i w_i = 1$. We enforce the simplex constraint via softmax normalization of the raw actor output.
- **Reward** $r_t$: The portfolio log-return minus a transaction cost penalty:

$$r_t = \log\left(\sum_{i=1}^{N} w_{t,i} \cdot \frac{p_{t+1,i}}{p_{t,i}}\right) - \lambda \sum_{i=1}^{N} |w_{t,i} - w_{t-1,i}|$$

where $\lambda$ is a transaction cost coefficient.

### Multi-Asset Considerations

For a portfolio with $N$ assets, the action space is $N$-dimensional and continuous. TD3 handles this naturally since DDPG-family algorithms are designed for continuous action spaces. The key design decisions include:

1. **Feature engineering**: Use rolling windows of returns (5, 10, 20 periods), volatility, volume ratio, and correlation features.
2. **Normalization**: Z-score normalize all features using a rolling window to handle non-stationarity.
3. **Action post-processing**: Apply softmax to the raw network output to ensure valid portfolio weights.
4. **Exploration**: Add Gaussian noise to the actor output during training, then apply softmax.

### Why Not Discrete Actions?

Traditional DQN-based approaches discretize the weight space (e.g., 0%, 25%, 50%, 75%, 100% for each asset). For $N$ assets with $K$ discretization levels, the action space grows as $K^N$. With 5 assets and 20 weight levels, that is $20^5 = 3.2$ million actions. TD3 avoids this combinatorial explosion entirely.

---

## Rust Implementation

Our Rust implementation provides:

- **`Actor`**: A simple feedforward network producing deterministic portfolio weights via softmax.
- **`TwinCritic`**: Two independent critic networks, each mapping (state, action) to a scalar Q-value.
- **`TD3Agent`**: Orchestrates training with clipped double-Q targets, delayed policy updates, and target policy smoothing.
- **`ReplayBuffer`**: Experience storage with uniform random sampling.
- **`BybitClient`**: Fetches historical kline data from the Bybit public API.

The implementation uses `ndarray` for tensor operations and avoids heavy ML framework dependencies for transparency and educational value. All matrix operations are explicit, making it easy to trace the math from the equations above into the code.

Key implementation details:
- Network weights are initialized with Xavier/Glorot uniform initialization.
- Soft target updates use $\tau = 0.005$.
- Target policy noise $\sigma = 0.2$ with clip bound $c = 0.5$.
- Actor updates every $d = 2$ critic steps.
- Replay buffer capacity of 100,000 transitions with batch size 64.

---

## Bybit Data Integration

The implementation fetches OHLCV kline data from Bybit's public REST API (`/v5/market/kline`). We retrieve data for multiple trading pairs (e.g., BTCUSDT, ETHUSDT) and compute:

- Log returns: $r_t = \log(p_t / p_{t-1})$
- Rolling volatility over configurable windows
- Volume-weighted features

The data pipeline handles:
- Rate limiting and error recovery
- Timestamp alignment across multiple assets
- Missing data interpolation
- Feature normalization

---

## Key Takeaways

1. **TD3 is DDPG done right**: The three modifications (twin critics, delayed updates, target smoothing) are simple but dramatically improve stability and performance.

2. **Overestimation bias is dangerous in finance**: Unlike game environments where overestimation leads to suboptimal play, in trading it leads to real capital loss. TD3's conservative Q-estimates are a natural fit.

3. **Continuous action spaces suit portfolio management**: TD3 natively handles the continuous portfolio weight allocation problem without discretization, making it scalable to many assets.

4. **Delayed updates reduce churn**: By updating the actor less frequently, TD3 produces smoother portfolio weight trajectories, which translates directly to lower transaction costs.

5. **Target smoothing improves robustness**: Smoothing the target policy prevents overfitting to noise in the Q-function, which is critical when training on noisy financial data.

6. **Rust provides performance and safety**: The implementation demonstrates that TD3 can be built efficiently in Rust with explicit tensor operations, suitable for low-latency trading systems.

7. **Start simple, add complexity**: The modular design allows starting with a two-asset portfolio (BTC + ETH) and scaling to more assets by simply expanding the state and action dimensions.

---

## References

- Fujimoto, S., Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *ICML*.
- Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement learning. *ICLR*.
- Jiang, Z., Xu, D., & Liang, J. (2017). A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem. *arXiv*.
- Silver, D., et al. (2014). Deterministic Policy Gradient Algorithms. *ICML*.
