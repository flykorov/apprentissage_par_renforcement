import numpy as np
import pickle
import random


class Q_learning:
    def __init__(
        self,
        actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.95,
        episodes=5000,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes

        self.actions = actions
        self.n_actions = len(actions)
        self.q_table = {}

    def _q_values(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(self.n_actions)
        return self.q[state]

    def choose_action(self, state, greedy=False):
        """Politique epsilon-greedy : la plupart du temps on prend la
        meilleure action connue (exploitation), mais avec une probabilité
        epsilon on en essaie une au hasard (exploration) -- sans ça, l'agent
        ne découvrirait jamais rien de mieux que sa toute première idée.
        greedy=True force à toujours prendre la meilleure (utilisé une fois
        l'entraînement terminé, pour évaluer/jouer)."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self._q_values(state)))

    def update(self, state, action, reward, next_state, done):
        q_sa = self._q_values(state)
        if done:
            target = reward  # pas de "futur" après un état terminal
        else:
            target = reward + self.gamma * np.max(self._q_values(next_state))
        q_sa[action] += self.alpha * (target - q_sa[action])

    def decay_epsilon(self):
        """À appeler une fois par épisode : on explore beaucoup au début,
        de moins en moins au fil de l'entraînement."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.q = pickle.load(f)

    def __len__(self):
        """Nombre d'états distincts déjà rencontrés -- pratique pour suivre
        la taille de la Q-table pendant l'entraînement."""
        return len(self.q)
