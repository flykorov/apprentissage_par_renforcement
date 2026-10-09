# Circuit de course A -> B - Génération procédurale (Étape 1 du projet RL)

## Contenu
- `track_generator.py` : génération procédurale d'un circuit OUVERT (point A
  -> point B, pas de boucle) avec virages, ligne de départ, ligne d'arrivée,
  checkpoints, et détection "sur piste / hors piste".
- `car.py` : voiture avec physique simple. Gère le crash (sortie de piste =
  définitivement hors d'usage, ne répond plus aux commandes) et la victoire
  (ligne d'arrivée franchie).
- `main.py` : boucle pygame pour piloter manuellement la voiture et valider
  le circuit, avec affichage CRASH / VICTOIRE.

## Installation
```
pip install -r requirements.txt
python main.py
```

## Commandes
- Flèches ou ZQSD/WASD : conduire
- R : nouveau circuit (nouveau seed aléatoire)
- ESPACE : rejouer le même circuit (après un crash ou une victoire)
- Echap : quitter

## Comment ça marche (génération procédurale)
1. Un point de départ (à gauche) et un point d'arrivée (à droite) sont tirés.
2. Des points intermédiaires sont placés avec une coordonnée, projetée sur
   l'axe départ->arrivée, strictement croissante : ça garantit que le
   circuit ne peut jamais revenir en arrière sur lui-même. Le décalage
   latéral aléatoire de chaque point crée les virages.
3. Lissage par spline de Catmull-Rom (ouverte, pas de bouclage).
4. Offset de la centerline (+/- largeur/2) -> bord gauche / bord droit.
5. Vérification anti-auto-intersection des bords (régénère automatiquement
   si un virage est trop serré par rapport à la largeur de la piste).
6. Premier point = ligne de départ, dernier point = ligne d'arrivée.
   Des checkpoints réguliers sont générés le long du circuit.

## Règles du jeu
- **Crash** : dès que le centre de la voiture sort du ruban de la piste
  (`Track.point_on_track`), la voiture est marquée `crashed = True` : elle
  s'arrête net et n'écoute plus aucune touche. Elle est rendue en gris avec
  une croix. Il faut appuyer sur ESPACE ou R pour reprendre.
- **Victoire** : dès que la voiture est à moins d'une demi-largeur de piste
  du point d'arrivée (`Track.distance_to_finish`), elle est marquée
  `finished = True` : victoire, temps affiché.

`Track.point_on_track((x, y))` et `Track.distance_to_finish((x, y))` sont les
deux briques de base pour construire la fonction de récompense d'un futur
environnement RL (pénalité forte si hors piste / fin d'épisode, récompense à
l'arrivée, récompense de progression via les checkpoints).

## Prochaines étapes du projet (de A à Z)
1. ✅ Génération procédurale du circuit A->B + conduite manuelle + crash +
   victoire (fait).
2. Capteurs de la voiture (rayons de distance jusqu'aux bords) -> état pour
   l'agent RL.
3. Environnement Gymnasium (`reset()`, `step()`, reward basé sur les
   checkpoints + la distance à l'arrivée + pénalité de crash).
4. Entraînement d'un agent (ex: PPO/DQN via stable-baselines3).
5. Visualisation de l'agent entraîné + comparaison sur plusieurs circuits.
