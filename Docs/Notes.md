# Notes Trading Bot

On crée un environnement, et notre propre Agent, car les librairies comme Stable_baseline3 n'ont pas d'architecture DQN avec LSTM.

Steps Agent & Environnement : 
- L’environnement reçoit une action et renvoie un état, une récompense et un indicateur de fin d’épisode.
- L’agent décide quelle action prendre en fonction de l’état (select_action).
- Il stocke la transition (store_transition).
- Il entraîne son réseau de neurones (train_step).
- Il ajuste progressivement sa politique d’exploration (update_epsilon).
- Il synchronise périodiquement son réseau cible (update_target).

**store_transition** : Contrairement au Q-learning classique, on n'entraîne pas le réseau immédiatement après chaque action.
Si on entraînait le réseau à chaque interaction, les corrélations temporelles entre états successifs créeraient un biais.
Solution : Utiliser un Replay Buffer pour stocker des transitions et s’entraîner sur un échantillon aléatoire plus tard.
Avantage : Cela permet un apprentissage plus stable et évite les oscillations dans les poids du réseau.

**train_step** : On entraîne le réseau en différé, ce qui réduit la corrélation entre transitions consécutives.

**update_epsilon** : Diminue progressivement la probabilité d’explorer (ε-greedy).

**update_target** : Lorsqu’on entraîne un réseau de neurones, ses valeurs changent constamment.
Dans DQN, la cible de mise à jour est basée sur le même réseau, ce qui peut le rendre instable.
Solution : Utiliser un réseau cible fixe pendant plusieurs étapes, puis le mettre à jour périodiquement.

**get_obs** : 
- self.position_value : La valeur actuelle de la position ouverte (si l'agent a acheté une action, cela indique le gain/perte latent).
- self.history : L'historique des variations de prix (différences entre les prix de clôture des jours précédents).
- feats : Les indicateurs de marché actuels :
    - Close(t): Prix de clôture actuel.
    - RSI(t): Indice de force relative (RSI), avec une gestion des valeurs NaN.
    - EMA12(t): Moyenne mobile exponentielle sur 12 périodes.

**select_action** : 
- Exploration (epsilon-greedy) : Avec une probabilité de epsilon, l'agent choisit une action aléatoire, favorisant ainsi l'exploration.
- Exploitation : Si l'agent ne choisit pas une action aléatoire, il utilise son modèle (réseau de neurones) pour sélectionner l'action qui a la Q-value la plus élevée parmi toutes les actions possibles, ce qui lui permet de maximiser ses récompenses à long terme.

Equation dans train step tiré de l'**équation de Bellman** : Q(s,a)=E[R_t + γ ⋅ a′maxQ(s′,a′)]

En faisant un Train/Test split avec Shuffle on obteint des meilleures résultats



# A faire
- Train et test sur plusieurs années
- Augmenter le temps de retiens avant màj du NN (5ans dans le papier)
- Construire un NN avec des LSTM
- Leaky layer
- Faire un plot avec différent coûts de transaction

# En plus
- implémenter un agent plus complexe, avec un réseau plus profond, et des features plus avancées.
- optimiser les hyperparamètres (grid search, etc.)
- ajouter des features (volatilité, etc.)
- ajouter une manière d'acheter plusieurs action à la fois (quantité variable)
- ajouter des frais de transaction
- Augmenter la fréquence de trading (intraday) et de points
- ajouter un stop-loss sur les positions
- ajouter un take-profit
- ajouter un système de récompense plus complexe
- ajouter une courbe de performance (capital cumulé)
- ajouter une visualisation des trades (graphique)
- ajouter une visualisation des rewards (graphique)
- ajouter une visualisation des pertes (graphique)
- ajouter un système de logging (TensorBoard, etc.)
- ajouter une sauvegarde du modèle
- ajouter un système de validation (early stopping)
- ajouter un système de rendu vidéo (Gym)
- essayer sur d'autres actifs (forex, crypto, etc.)
- essayer sur d'autres environnements (Gym)
- essayer d'autres algorithmes (PPO, DDPG, etc.)
- essayer d'autres techniques de deep learning (LSTM, etc.)
- essayer d'autres techniques de reinforcement learning (DQN, etc.)
- essayer d'autres techniques de trading (pairs trading, etc.)
- CNN dont la loss est f(xi) - somme des différences croissantes
- Rajouter le volume dans les features
- Rajouter des features de volatilité
- Rajouter des features de momentum
- Rajouter des features de tendance
- Rajouter des features de support et résistance
- Rajouter des features de gap
- Rajouter des features de chandeliers japonais
- Rajouter des features de pattern recognition
- Rajouter des features de news sentiment
- Rajouter des features de sentiment analysis
- Rajouter des features de macroéconomie
- Rajouter des features de microéconomie
- Rajouter des features de taux d'intérêt


# Fait
- Utiliser des métriques de performances pour le modèle [Ref](https://arxiv.org/pdf/1904.04912) :
    1. **E(R)**: annualised expected trade return,
    2. **std(R)**: annualised standard deviation of trade return,
    3. **Downside Deviation (DD)**: annualised standard deviation of trade returns that are negative, also known as downside risk,
    4. **Sharpe**: annualised Sharpe Ratio (E(R)/std(R)),
    5. **Sortino**: a variant of Sharpe Ratio that uses downside deviation as risk measures (E(R)/Downside Deviation),
    6. **MDD**: maximum drawdown shows the maximum observed loss from any peak of a portfolio,
    7. **Calmar**: the Calmar ratio compares the expected annual rate of return with maximum drawdown. In general, the higher the ratio is, the better the performance of the portfolio is,
    8. **% +ve Returns**: percentage of positive trade returns,
    9. **Ave. P/Ave. L** : the ratio between positive and negative trade returns.

- Vérifier frais de transactions dans le calcul des métriques

