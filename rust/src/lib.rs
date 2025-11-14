//! # TD3 Portfolio Trading
//!
//! Implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3)
//! for continuous portfolio rebalancing with Bybit market data.

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ─── Linear Layer ───────────────────────────────────────────────────────────

/// A single fully-connected layer: y = x * W + b
#[derive(Clone, Debug)]
pub struct Linear {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
}

impl Linear {
    /// Xavier uniform initialization.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (input_dim + output_dim) as f64).sqrt();
        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-limit..limit)
        });
        let bias = Array1::zeros(output_dim);
        Self { weights, bias }
    }

    /// Forward pass for a single sample.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        input.dot(&self.weights) + &self.bias
    }
}

// ─── Activation helpers ─────────────────────────────────────────────────────

fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps = x.mapv(|v| (v - max_val).exp());
    let sum: f64 = exps.sum();
    if sum > 0.0 {
        exps / sum
    } else {
        Array1::from_elem(x.len(), 1.0 / x.len() as f64)
    }
}

// ─── Actor Network ──────────────────────────────────────────────────────────

/// Deterministic policy network that outputs portfolio weights via softmax.
#[derive(Clone, Debug)]
pub struct Actor {
    pub layer1: Linear,
    pub layer2: Linear,
    pub layer3: Linear,
    pub state_dim: usize,
    pub action_dim: usize,
}

impl Actor {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self {
            layer1: Linear::new(state_dim, hidden_dim),
            layer2: Linear::new(hidden_dim, hidden_dim),
            layer3: Linear::new(hidden_dim, action_dim),
            state_dim,
            action_dim,
        }
    }

    /// Forward pass: state -> portfolio weights (summing to 1).
    pub fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        let h1 = relu(&self.layer1.forward(state));
        let h2 = relu(&self.layer2.forward(&h1));
        let out = self.layer3.forward(&h2);
        softmax(&out)
    }

    /// Copy parameters from another actor.
    pub fn copy_from(&mut self, other: &Actor) {
        self.layer1.weights.assign(&other.layer1.weights);
        self.layer1.bias.assign(&other.layer1.bias);
        self.layer2.weights.assign(&other.layer2.weights);
        self.layer2.bias.assign(&other.layer2.bias);
        self.layer3.weights.assign(&other.layer3.weights);
        self.layer3.bias.assign(&other.layer3.bias);
    }

    /// Soft update: self = tau * source + (1-tau) * self
    pub fn soft_update(&mut self, source: &Actor, tau: f64) {
        macro_rules! blend {
            ($dst:expr, $src:expr) => {
                $dst.zip_mut_with(&$src, |d, &s| *d = tau * s + (1.0 - tau) * *d);
            };
        }
        blend!(self.layer1.weights, source.layer1.weights);
        blend!(self.layer1.bias, source.layer1.bias);
        blend!(self.layer2.weights, source.layer2.weights);
        blend!(self.layer2.bias, source.layer2.bias);
        blend!(self.layer3.weights, source.layer3.weights);
        blend!(self.layer3.bias, source.layer3.bias);
    }
}

// ─── Critic Network ─────────────────────────────────────────────────────────

/// Single Q-network: (state, action) -> scalar Q-value.
#[derive(Clone, Debug)]
pub struct Critic {
    pub layer1: Linear,
    pub layer2: Linear,
    pub layer3: Linear,
}

impl Critic {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self {
            layer1: Linear::new(state_dim + action_dim, hidden_dim),
            layer2: Linear::new(hidden_dim, hidden_dim),
            layer3: Linear::new(hidden_dim, 1),
        }
    }

    /// Forward pass: concatenate state and action, output scalar Q-value.
    pub fn forward(&self, state: &Array1<f64>, action: &Array1<f64>) -> f64 {
        let mut input = Array1::zeros(state.len() + action.len());
        input.slice_mut(ndarray::s![..state.len()]).assign(state);
        input
            .slice_mut(ndarray::s![state.len()..])
            .assign(action);

        let h1 = relu(&self.layer1.forward(&input));
        let h2 = relu(&self.layer2.forward(&h1));
        let out = self.layer3.forward(&h2);
        out[0]
    }

    /// Copy parameters from another critic.
    pub fn copy_from(&mut self, other: &Critic) {
        self.layer1.weights.assign(&other.layer1.weights);
        self.layer1.bias.assign(&other.layer1.bias);
        self.layer2.weights.assign(&other.layer2.weights);
        self.layer2.bias.assign(&other.layer2.bias);
        self.layer3.weights.assign(&other.layer3.weights);
        self.layer3.bias.assign(&other.layer3.bias);
    }

    /// Soft update: self = tau * source + (1-tau) * self
    pub fn soft_update(&mut self, source: &Critic, tau: f64) {
        macro_rules! blend {
            ($dst:expr, $src:expr) => {
                $dst.zip_mut_with(&$src, |d, &s| *d = tau * s + (1.0 - tau) * *d);
            };
        }
        blend!(self.layer1.weights, source.layer1.weights);
        blend!(self.layer1.bias, source.layer1.bias);
        blend!(self.layer2.weights, source.layer2.weights);
        blend!(self.layer2.bias, source.layer2.bias);
        blend!(self.layer3.weights, source.layer3.weights);
        blend!(self.layer3.bias, source.layer3.bias);
    }
}

// ─── Twin Critic ────────────────────────────────────────────────────────────

/// Two independent critics for clipped double-Q learning.
#[derive(Clone, Debug)]
pub struct TwinCritic {
    pub q1: Critic,
    pub q2: Critic,
}

impl TwinCritic {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self {
            q1: Critic::new(state_dim, action_dim, hidden_dim),
            q2: Critic::new(state_dim, action_dim, hidden_dim),
        }
    }

    /// Return the minimum of the two Q-value estimates (clipped double-Q).
    pub fn min_q(&self, state: &Array1<f64>, action: &Array1<f64>) -> f64 {
        let q1_val = self.q1.forward(state, action);
        let q2_val = self.q2.forward(state, action);
        q1_val.min(q2_val)
    }

    pub fn copy_from(&mut self, other: &TwinCritic) {
        self.q1.copy_from(&other.q1);
        self.q2.copy_from(&other.q2);
    }

    pub fn soft_update(&mut self, source: &TwinCritic, tau: f64) {
        self.q1.soft_update(&source.q1, tau);
        self.q2.soft_update(&source.q2, tau);
    }
}

// ─── Replay Buffer ──────────────────────────────────────────────────────────

/// A transition stored in the replay buffer.
#[derive(Clone, Debug)]
pub struct Transition {
    pub state: Array1<f64>,
    pub action: Array1<f64>,
    pub reward: f64,
    pub next_state: Array1<f64>,
    pub done: bool,
}

/// Uniform replay buffer with fixed capacity.
pub struct ReplayBuffer {
    pub storage: Vec<Transition>,
    pub capacity: usize,
    pub position: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            storage: Vec::with_capacity(capacity),
            capacity,
            position: 0,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.storage.len() < self.capacity {
            self.storage.push(transition);
        } else {
            self.storage[self.position] = transition;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        let mut rng = rand::thread_rng();
        let len = self.storage.len();
        (0..batch_size)
            .map(|_| &self.storage[rng.gen_range(0..len)])
            .collect()
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }
}

// ─── TD3 Agent ──────────────────────────────────────────────────────────────

/// Hyperparameters for TD3.
pub struct TD3Config {
    pub gamma: f64,
    pub tau: f64,
    pub policy_noise: f64,
    pub noise_clip: f64,
    pub policy_delay: usize,
    pub actor_lr: f64,
    pub critic_lr: f64,
    pub batch_size: usize,
}

impl Default for TD3Config {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            tau: 0.005,
            policy_noise: 0.2,
            noise_clip: 0.5,
            policy_delay: 2,
            actor_lr: 0.001,
            critic_lr: 0.001,
            batch_size: 64,
        }
    }
}

/// TD3 agent for portfolio trading.
pub struct TD3Agent {
    pub actor: Actor,
    pub actor_target: Actor,
    pub critic: TwinCritic,
    pub critic_target: TwinCritic,
    pub config: TD3Config,
    pub total_steps: usize,
    pub action_dim: usize,
}

impl TD3Agent {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize, config: TD3Config) -> Self {
        let actor = Actor::new(state_dim, action_dim, hidden_dim);
        let mut actor_target = Actor::new(state_dim, action_dim, hidden_dim);
        actor_target.copy_from(&actor);

        let critic = TwinCritic::new(state_dim, action_dim, hidden_dim);
        let mut critic_target = TwinCritic::new(state_dim, action_dim, hidden_dim);
        critic_target.copy_from(&critic);

        Self {
            actor,
            actor_target,
            critic,
            critic_target,
            config,
            total_steps: 0,
            action_dim,
        }
    }

    /// Select action with exploration noise.
    pub fn select_action(&self, state: &Array1<f64>, exploration_noise: f64) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let action = self.actor.forward(state);

        // Add exploration noise and re-normalize via softmax
        let noisy: Array1<f64> = action.mapv(|a| {
            let noise: f64 = rng.gen_range(-exploration_noise..exploration_noise);
            (a + noise).max(0.0)
        });

        // Re-normalize to sum to 1
        let sum: f64 = noisy.sum();
        if sum > 0.0 {
            noisy / sum
        } else {
            Array1::from_elem(self.action_dim, 1.0 / self.action_dim as f64)
        }
    }

    /// Select action without noise (for evaluation).
    pub fn select_action_deterministic(&self, state: &Array1<f64>) -> Array1<f64> {
        self.actor.forward(state)
    }

    /// Compute target actions with policy smoothing (clipped noise).
    pub fn target_action_with_smoothing(&self, next_state: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let target_action = self.actor_target.forward(next_state);

        let smoothed: Array1<f64> = target_action.mapv(|a| {
            let noise: f64 = rng
                .gen::<f64>()
                .mul_add(2.0 * self.config.policy_noise, -self.config.policy_noise)
                .max(-self.config.noise_clip)
                .min(self.config.noise_clip);
            (a + noise).max(0.0).min(1.0)
        });

        // Re-normalize
        let sum: f64 = smoothed.sum();
        if sum > 0.0 {
            smoothed / sum
        } else {
            Array1::from_elem(self.action_dim, 1.0 / self.action_dim as f64)
        }
    }

    /// Perform one training step on a batch from the replay buffer.
    ///
    /// Returns (critic_loss, actor_updated) for monitoring.
    pub fn train_step(&mut self, buffer: &ReplayBuffer) -> (f64, bool) {
        if buffer.len() < self.config.batch_size {
            return (0.0, false);
        }

        self.total_steps += 1;
        let batch = buffer.sample(self.config.batch_size);

        // Compute target Q-values with policy smoothing
        let mut critic_loss = 0.0;

        for transition in &batch {
            let target_action = self.target_action_with_smoothing(&transition.next_state);
            let target_q = self
                .critic_target
                .min_q(&transition.next_state, &target_action);

            let done_mask = if transition.done { 0.0 } else { 1.0 };
            let y = transition.reward + self.config.gamma * done_mask * target_q;

            let q1 = self
                .critic
                .q1
                .forward(&transition.state, &transition.action);
            let q2 = self
                .critic
                .q2
                .forward(&transition.state, &transition.action);

            // Accumulate loss for monitoring (actual gradient updates are simplified)
            critic_loss += (q1 - y).powi(2) + (q2 - y).powi(2);

            // Simplified gradient update for critic (finite-difference approximation)
            self.update_critic_weights(&transition.state, &transition.action, y);
        }

        critic_loss /= (2 * batch.len()) as f64;

        // Delayed policy update
        let actor_updated = self.total_steps % self.config.policy_delay == 0;
        if actor_updated {
            for transition in &batch {
                self.update_actor_weights(&transition.state);
            }

            // Soft update target networks
            self.actor_target
                .soft_update(&self.actor, self.config.tau);
            self.critic_target
                .soft_update(&self.critic, self.config.tau);
        }

        (critic_loss, actor_updated)
    }

    /// Simplified critic weight update using finite-difference gradient.
    fn update_critic_weights(&mut self, state: &Array1<f64>, action: &Array1<f64>, target: f64) {
        let lr = self.config.critic_lr;
        let epsilon = 1e-4;

        // Update Q1 weights (layer3 only for efficiency)
        for i in 0..self.critic.q1.layer3.weights.nrows() {
            for j in 0..self.critic.q1.layer3.weights.ncols() {
                let original = self.critic.q1.layer3.weights[[i, j]];

                self.critic.q1.layer3.weights[[i, j]] = original + epsilon;
                let loss_plus = (self.critic.q1.forward(state, action) - target).powi(2);

                self.critic.q1.layer3.weights[[i, j]] = original - epsilon;
                let loss_minus = (self.critic.q1.forward(state, action) - target).powi(2);

                self.critic.q1.layer3.weights[[i, j]] = original;

                let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
                self.critic.q1.layer3.weights[[i, j]] -= lr * grad;
            }
        }

        // Update Q2 weights (layer3 only for efficiency)
        for i in 0..self.critic.q2.layer3.weights.nrows() {
            for j in 0..self.critic.q2.layer3.weights.ncols() {
                let original = self.critic.q2.layer3.weights[[i, j]];

                self.critic.q2.layer3.weights[[i, j]] = original + epsilon;
                let loss_plus = (self.critic.q2.forward(state, action) - target).powi(2);

                self.critic.q2.layer3.weights[[i, j]] = original - epsilon;
                let loss_minus = (self.critic.q2.forward(state, action) - target).powi(2);

                self.critic.q2.layer3.weights[[i, j]] = original;

                let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
                self.critic.q2.layer3.weights[[i, j]] -= lr * grad;
            }
        }
    }

    /// Simplified actor weight update: maximize Q1(s, actor(s)).
    fn update_actor_weights(&mut self, state: &Array1<f64>) {
        let lr = self.config.actor_lr;
        let epsilon = 1e-4;

        // Update actor layer3 weights to maximize Q1
        for i in 0..self.actor.layer3.weights.nrows() {
            for j in 0..self.actor.layer3.weights.ncols() {
                let original = self.actor.layer3.weights[[i, j]];

                self.actor.layer3.weights[[i, j]] = original + epsilon;
                let action_plus = self.actor.forward(state);
                let q_plus = self.critic.q1.forward(state, &action_plus);

                self.actor.layer3.weights[[i, j]] = original - epsilon;
                let action_minus = self.actor.forward(state);
                let q_minus = self.critic.q1.forward(state, &action_minus);

                self.actor.layer3.weights[[i, j]] = original;

                // Gradient ascent (maximize Q)
                let grad = (q_plus - q_minus) / (2.0 * epsilon);
                self.actor.layer3.weights[[i, j]] += lr * grad;
            }
        }
    }
}

// ─── Bybit API Client ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// OHLCV candle parsed from Bybit API.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Client for fetching market data from Bybit public API.
pub struct BybitClient {
    pub base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data for a symbol.
    ///
    /// - `symbol`: e.g. "BTCUSDT"
    /// - `interval`: e.g. "60" for 1-hour candles
    /// - `limit`: number of candles (max 200)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitKlineResponse = reqwest::blocking::get(&url)?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error {}: {}",
                response.ret_code,
                response.ret_msg
            );
        }

        let candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Candle {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(candles)
    }
}

// ─── Feature Engineering ────────────────────────────────────────────────────

/// Compute log returns from a series of closing prices.
pub fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Compute rolling volatility (standard deviation of returns).
pub fn compute_rolling_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.len() < window {
        return vec![0.0; returns.len()];
    }

    let mut result = vec![0.0; window - 1];

    for i in (window - 1)..returns.len() {
        let slice = &returns[(i + 1 - window)..=i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let variance: f64 = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window as f64;
        result.push(variance.sqrt());
    }

    result
}

/// Build a state vector from multi-asset price data.
/// State includes: returns, volatilities, and current portfolio weights for each asset.
pub fn build_state(
    returns_per_asset: &[Vec<f64>],
    volatilities_per_asset: &[Vec<f64>],
    current_weights: &[f64],
    time_index: usize,
) -> Array1<f64> {
    let mut state = Vec::new();

    for (returns, vols) in returns_per_asset.iter().zip(volatilities_per_asset.iter()) {
        if time_index < returns.len() {
            state.push(returns[time_index]);
        } else {
            state.push(0.0);
        }
        if time_index < vols.len() {
            state.push(vols[time_index]);
        } else {
            state.push(0.0);
        }
    }

    state.extend_from_slice(current_weights);

    Array1::from_vec(state)
}

/// Compute portfolio reward: log return minus transaction cost.
pub fn compute_reward(
    weights: &[f64],
    prev_weights: &[f64],
    returns: &[f64],
    transaction_cost: f64,
) -> f64 {
    // Portfolio return
    let port_return: f64 = weights
        .iter()
        .zip(returns.iter())
        .map(|(w, r)| w * r.exp())
        .sum::<f64>()
        .ln();

    // Transaction cost penalty
    let turnover: f64 = weights
        .iter()
        .zip(prev_weights.iter())
        .map(|(w, pw)| (w - pw).abs())
        .sum();

    port_return - transaction_cost * turnover
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer_dimensions() {
        let layer = Linear::new(10, 5);
        assert_eq!(layer.weights.shape(), &[10, 5]);
        assert_eq!(layer.bias.len(), 5);

        let input = Array1::zeros(10);
        let output = layer.forward(&input);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_actor_output_sums_to_one() {
        let actor = Actor::new(8, 3, 32);
        let state = Array1::from_vec(vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 0.5, 0.5]);
        let action = actor.forward(&state);

        assert_eq!(action.len(), 3);
        let sum: f64 = action.sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Portfolio weights should sum to 1, got {}",
            sum
        );
        assert!(action.iter().all(|&w| w >= 0.0), "Weights must be non-negative");
    }

    #[test]
    fn test_twin_critic_min_q() {
        let tc = TwinCritic::new(8, 3, 32);
        let state = Array1::from_vec(vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 0.5, 0.5]);
        let action = Array1::from_vec(vec![0.4, 0.3, 0.3]);

        let q1 = tc.q1.forward(&state, &action);
        let q2 = tc.q2.forward(&state, &action);
        let min_q = tc.min_q(&state, &action);

        assert_eq!(min_q, q1.min(q2));
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);
        assert!(buffer.is_empty());

        for i in 0..50 {
            buffer.push(Transition {
                state: Array1::from_vec(vec![i as f64]),
                action: Array1::from_vec(vec![0.5]),
                reward: 1.0,
                next_state: Array1::from_vec(vec![(i + 1) as f64]),
                done: false,
            });
        }

        assert_eq!(buffer.len(), 50);
        let batch = buffer.sample(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_replay_buffer_overflow() {
        let mut buffer = ReplayBuffer::new(5);

        for i in 0..10 {
            buffer.push(Transition {
                state: Array1::from_vec(vec![i as f64]),
                action: Array1::from_vec(vec![0.5]),
                reward: 1.0,
                next_state: Array1::from_vec(vec![(i + 1) as f64]),
                done: false,
            });
        }

        // Buffer should not exceed capacity
        assert_eq!(buffer.len(), 5);
    }

    #[test]
    fn test_soft_update() {
        let actor1 = Actor::new(4, 2, 16);
        let mut actor2 = Actor::new(4, 2, 16);

        let original_w = actor2.layer1.weights.clone();
        actor2.soft_update(&actor1, 0.005);

        // Weights should have changed (very slightly)
        let diff: f64 = (&actor2.layer1.weights - &original_w)
            .mapv(f64::abs)
            .sum();
        // They should differ since actor1 and actor2 were independently initialized
        assert!(diff > 0.0, "Soft update should change weights");
    }

    #[test]
    fn test_target_policy_smoothing() {
        let agent = TD3Agent::new(8, 3, 32, TD3Config::default());
        let state = Array1::from_vec(vec![0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 0.5, 0.5]);

        let action = agent.target_action_with_smoothing(&state);
        assert_eq!(action.len(), 3);

        let sum: f64 = action.sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Smoothed action should sum to 1, got {}",
            sum
        );
        assert!(
            action.iter().all(|&w| w >= 0.0),
            "Smoothed weights must be non-negative"
        );
    }

    #[test]
    fn test_delayed_policy_update() {
        let mut agent = TD3Agent::new(4, 2, 16, TD3Config::default());
        let mut buffer = ReplayBuffer::new(1000);

        // Fill buffer with random transitions
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            buffer.push(Transition {
                state: Array1::from_vec(vec![
                    rng.gen::<f64>(),
                    rng.gen::<f64>(),
                    rng.gen::<f64>(),
                    rng.gen::<f64>(),
                ]),
                action: Array1::from_vec(vec![0.5, 0.5]),
                reward: rng.gen::<f64>() - 0.5,
                next_state: Array1::from_vec(vec![
                    rng.gen::<f64>(),
                    rng.gen::<f64>(),
                    rng.gen::<f64>(),
                    rng.gen::<f64>(),
                ]),
                done: false,
            });
        }

        // Step 1: critic updates, no actor update
        let (_, actor_updated_1) = agent.train_step(&buffer);
        assert!(!actor_updated_1, "Actor should not update on step 1");

        // Step 2: critic + actor update (policy_delay=2)
        let (_, actor_updated_2) = agent.train_step(&buffer);
        assert!(actor_updated_2, "Actor should update on step 2");
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 105.0, 103.0, 110.0];
        let returns = compute_log_returns(&prices);
        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_volatility() {
        let returns = vec![0.01, -0.02, 0.03, 0.01, -0.01, 0.02];
        let vol = compute_rolling_volatility(&returns, 3);
        assert_eq!(vol.len(), returns.len());
        // First two should be zero (not enough data)
        assert_eq!(vol[0], 0.0);
        assert_eq!(vol[1], 0.0);
        // Third onward should be positive
        assert!(vol[2] > 0.0);
    }

    #[test]
    fn test_compute_reward() {
        let weights = vec![0.6, 0.4];
        let prev_weights = vec![0.5, 0.5];
        let returns = vec![0.01, -0.005];
        let reward = compute_reward(&weights, &prev_weights, &returns, 0.001);

        // Should be a finite number
        assert!(reward.is_finite());

        // With zero transaction cost and positive returns, reward should be positive
        let reward_no_cost = compute_reward(&weights, &prev_weights, &returns, 0.0);
        assert!(reward_no_cost > 0.0);
    }
}
