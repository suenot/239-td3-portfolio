//! # TD3 Portfolio Trading Example
//!
//! Fetches BTC and ETH data from Bybit, trains a TD3 portfolio agent,
//! compares with a DDPG-like baseline (single critic), and shows portfolio
//! weights over time.

use ndarray::Array1;
use td3_portfolio::*;

/// Simple DDPG-like agent (single critic, no delay, no smoothing) for comparison.
#[allow(dead_code)]
struct DDPGBaseline {
    actor: Actor,
    critic: Critic,
    gamma: f64,
}

impl DDPGBaseline {
    fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self {
            actor: Actor::new(state_dim, action_dim, hidden_dim),
            critic: Critic::new(state_dim, action_dim, hidden_dim),
            gamma: 0.99,
        }
    }

    fn q_value(&self, state: &Array1<f64>, action: &Array1<f64>) -> f64 {
        self.critic.forward(state, action)
    }
}

fn main() -> anyhow::Result<()> {
    println!("=== TD3 Portfolio Trading Example ===\n");

    // ── Fetch data from Bybit ──────────────────────────────────────────────
    println!("Fetching market data from Bybit...");
    let client = BybitClient::new();

    let btc_candles = match client.fetch_klines("BTCUSDT", "60", 200) {
        Ok(c) => {
            println!("  Fetched {} BTC candles", c.len());
            c
        }
        Err(e) => {
            println!("  Could not fetch BTC data ({}), using synthetic data", e);
            generate_synthetic_candles(200, 50000.0, 0.02)
        }
    };

    let eth_candles = match client.fetch_klines("ETHUSDT", "60", 200) {
        Ok(c) => {
            println!("  Fetched {} ETH candles", c.len());
            c
        }
        Err(e) => {
            println!("  Could not fetch ETH data ({}), using synthetic data", e);
            generate_synthetic_candles(200, 3000.0, 0.025)
        }
    };

    // ── Prepare features ───────────────────────────────────────────────────
    let btc_prices: Vec<f64> = btc_candles.iter().map(|c| c.close).collect();
    let eth_prices: Vec<f64> = eth_candles.iter().map(|c| c.close).collect();

    let min_len = btc_prices.len().min(eth_prices.len());
    let btc_prices = &btc_prices[..min_len];
    let eth_prices = &eth_prices[..min_len];

    let btc_returns = compute_log_returns(btc_prices);
    let eth_returns = compute_log_returns(eth_prices);

    let btc_vol = compute_rolling_volatility(&btc_returns, 5);
    let eth_vol = compute_rolling_volatility(&eth_returns, 5);

    let returns_per_asset = vec![btc_returns.clone(), eth_returns.clone()];
    let vols_per_asset = vec![btc_vol, eth_vol];

    let num_assets = 2;
    // State: (return, vol) per asset + current weights = 2*2 + 2 = 6
    let state_dim = 2 * num_assets + num_assets;
    let action_dim = num_assets;
    let hidden_dim = 64;

    println!(
        "\nState dim: {}, Action dim: {}, Data points: {}\n",
        state_dim,
        action_dim,
        btc_returns.len()
    );

    // ── Initialize TD3 agent ───────────────────────────────────────────────
    let config = TD3Config {
        gamma: 0.99,
        tau: 0.005,
        policy_noise: 0.2,
        noise_clip: 0.5,
        policy_delay: 2,
        actor_lr: 0.0005,
        critic_lr: 0.001,
        batch_size: 32,
    };

    let mut agent = TD3Agent::new(state_dim, action_dim, hidden_dim, config);
    let mut buffer = ReplayBuffer::new(100_000);

    // DDPG baseline for comparison
    let ddpg = DDPGBaseline::new(state_dim, action_dim, hidden_dim);

    // ── Collect experience ─────────────────────────────────────────────────
    println!("Collecting experience and training...\n");

    let mut current_weights = vec![0.5, 0.5]; // Start with equal weights
    let mut portfolio_history: Vec<Vec<f64>> = Vec::new();
    let mut reward_history: Vec<f64> = Vec::new();
    let mut td3_q_values: Vec<f64> = Vec::new();
    let mut ddpg_q_values: Vec<f64> = Vec::new();
    let transaction_cost = 0.001;

    let data_len = btc_returns.len().min(eth_returns.len());
    let start_idx = 5; // Skip initial period for volatility warmup

    for t in start_idx..(data_len - 1) {
        let state = build_state(&returns_per_asset, &vols_per_asset, &current_weights, t);
        let action = agent.select_action(&state, 0.1);
        let new_weights: Vec<f64> = action.to_vec();

        // Step returns for this period
        let step_returns = vec![btc_returns[t], eth_returns[t]];
        let reward = compute_reward(&new_weights, &current_weights, &step_returns, transaction_cost);

        let next_state = build_state(&returns_per_asset, &vols_per_asset, &new_weights, t + 1);
        let done = t == data_len - 2;

        buffer.push(Transition {
            state: state.clone(),
            action: action.clone(),
            reward,
            next_state,
            done,
        });

        // Record Q-value estimates for comparison
        let td3_q = agent.critic.min_q(&state, &action);
        let ddpg_q = ddpg.q_value(&state, &action);
        td3_q_values.push(td3_q);
        ddpg_q_values.push(ddpg_q);

        // Train
        if buffer.len() >= 32 {
            agent.train_step(&buffer);
        }

        portfolio_history.push(new_weights.clone());
        reward_history.push(reward);
        current_weights = new_weights;
    }

    // ── Results ────────────────────────────────────────────────────────────
    println!("--- Training Results ---\n");

    let total_reward: f64 = reward_history.iter().sum();
    let avg_reward = total_reward / reward_history.len() as f64;
    println!("Total reward:   {:.6}", total_reward);
    println!("Average reward: {:.6}", avg_reward);
    println!("Episodes:       {}\n", reward_history.len());

    // Q-value comparison
    let avg_td3_q: f64 = td3_q_values.iter().sum::<f64>() / td3_q_values.len() as f64;
    let avg_ddpg_q: f64 = ddpg_q_values.iter().sum::<f64>() / ddpg_q_values.len() as f64;
    println!("--- Q-Value Comparison: TD3 vs DDPG ---\n");
    println!("Avg TD3  Q-value (min of twin): {:.6}", avg_td3_q);
    println!("Avg DDPG Q-value (single):      {:.6}", avg_ddpg_q);
    println!(
        "Difference (DDPG - TD3):        {:.6}",
        avg_ddpg_q - avg_td3_q
    );
    if avg_ddpg_q > avg_td3_q {
        println!("  -> DDPG overestimates by {:.4} on average (as expected)", avg_ddpg_q - avg_td3_q);
    }
    println!();

    // Portfolio weights over time
    println!("--- Portfolio Weights Over Time ---\n");
    println!("{:>5}  {:>10}  {:>10}", "Step", "BTC Weight", "ETH Weight");
    println!("{}", "-".repeat(30));

    let display_count = portfolio_history.len().min(20);
    let step_size = if portfolio_history.len() > 20 {
        portfolio_history.len() / 20
    } else {
        1
    };

    for i in 0..display_count {
        let idx = i * step_size;
        if idx < portfolio_history.len() {
            println!(
                "{:>5}  {:>10.4}  {:>10.4}",
                idx, portfolio_history[idx][0], portfolio_history[idx][1]
            );
        }
    }

    // Final allocation
    if let Some(last) = portfolio_history.last() {
        println!("\nFinal allocation: BTC={:.2}%, ETH={:.2}%",
            last[0] * 100.0, last[1] * 100.0);
    }

    println!("\n=== Done ===");
    Ok(())
}

/// Generate synthetic candle data for testing when API is unavailable.
fn generate_synthetic_candles(count: usize, start_price: f64, volatility: f64) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = start_price;
    let mut candles = Vec::with_capacity(count);

    for i in 0..count {
        let ret = rng.gen::<f64>() * 2.0 * volatility - volatility;
        let open = price;
        price *= 1.0 + ret;
        let close = price;
        let high = open.max(close) * (1.0 + rng.gen::<f64>() * 0.005);
        let low = open.min(close) * (1.0 - rng.gen::<f64>() * 0.005);
        let volume = rng.gen::<f64>() * 1000.0 + 100.0;

        candles.push(Candle {
            timestamp: 1700000000000 + (i as u64) * 3600000,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    candles
}
