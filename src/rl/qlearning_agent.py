"""
qlearning_agent.py — Reinforcement-learning price optimiser using Q-Learning.

Concept
-------
We model pricing as a single-step RL problem:

  • **State**   — the product's feature vector (demand predictors)
  • **Action**  — choose a price from a discrete grid [50, 55, …, 200]
  • **Reward**  — estimated revenue  =  price  ×  predicted_demand(price)

The Q-table maps (state_id, action_index) → expected reward.
After training for several episodes the agent picks the price that
maximises expected revenue.

Because the state space is continuous we discretise it with a simple
hash; for production you'd swap this for a Deep-Q Network (DQN).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from collections import defaultdict

from src.utils.config import (
    QL_PRICE_MIN, QL_PRICE_MAX, QL_PRICE_STEP,
    QL_EPISODES, QL_ALPHA, QL_GAMMA,
    QL_EPSILON_START, QL_EPSILON_MIN, QL_EPSILON_DECAY,
)


class QLearningPriceOptimizer:
    """
    Tabular Q-Learning agent that finds the price maximising revenue.

    Parameters
    ----------
    demand_fn : callable
        A function f(price, features) → predicted_demand.
        The agent uses it to compute reward = price × demand.
    features : np.ndarray
        1-D feature vector describing the product / context.
    """

    def __init__(self, demand_fn, features: np.ndarray):
        self.demand_fn = demand_fn
        self.features = features

        # Build the discrete action space (list of candidate prices)
        self.actions = list(
            range(QL_PRICE_MIN, QL_PRICE_MAX + 1, QL_PRICE_STEP)
        )
        self.n_actions = len(self.actions)

        # Q-table: defaultdict so unseen states start at 0
        self.q_table: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

        # Hyperparameters
        self.alpha = QL_ALPHA
        self.gamma = QL_GAMMA
        self.epsilon = QL_EPSILON_START
        self.epsilon_min = QL_EPSILON_MIN
        self.epsilon_decay = QL_EPSILON_DECAY

    def _state_key(self) -> str:
        """Discretise the continuous feature vector into a hashable key."""
        rounded = np.round(self.features, decimals=2)
        return str(rounded.tolist())

    def _get_reward(self, price: float) -> float:
        """Revenue = price × predicted demand at that price."""
        demand = self.demand_fn(price, self.features)
        return price * max(demand, 0)

    def _choose_action(self, state_key: str) -> int:
        """ε-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state_key]))

    def train(self, episodes: int = QL_EPISODES) -> None:
        """
        Run Q-Learning episodes.

        Each episode:
          1. Observe state → select action (price) → get reward
          2. Update Q-value using the Bellman equation
          3. Decay exploration rate
        """
        state_key = self._state_key()

        for ep in range(episodes):
            action_idx = self._choose_action(state_key)
            price = self.actions[action_idx]
            reward = self._get_reward(price)

            # Bellman update (single-step so next_max == 0)
            old_q = self.q_table[state_key][action_idx]
            self.q_table[state_key][action_idx] = (
                old_q + self.alpha * (reward - old_q)
            )

            # Decay epsilon
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
            )

        print(f"[ql_agent] Training done — {episodes} episodes, "
              f"final ε={self.epsilon:.4f}")

    def get_optimal_price(self) -> dict:
        """
        After training, return the best price and its expected revenue.

        Returns
        -------
        dict with optimal_price, expected_revenue, q_values
        """
        state_key = self._state_key()
        q_values = self.q_table[state_key]
        best_idx = int(np.argmax(q_values))
        optimal_price = self.actions[best_idx]
        expected_revenue = float(q_values[best_idx])

        return {
            "optimal_price": optimal_price,
            "expected_revenue": expected_revenue,
            "q_values": {
                self.actions[i]: float(q_values[i])
                for i in range(self.n_actions)
            },
        }

    def simulate_prices(self) -> list[dict]:
        """
        Evaluate every candidate price and return a table of
        price → demand → revenue useful for plotting.
        """
        results = []
        for price in self.actions:
            demand = self.demand_fn(price, self.features)
            revenue = price * max(demand, 0)
            results.append({
                "price": price,
                "predicted_demand": round(float(demand), 2),
                "expected_revenue": round(float(revenue), 2),
            })
        return results
