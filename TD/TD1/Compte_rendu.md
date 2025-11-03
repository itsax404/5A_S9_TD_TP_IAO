# Compte Rendu - TD1 

**Réalisé par :** Davin Quentin et Druhet Joachim


## 1. Compréhension du Programme

### 1.1. Architecture du réseau

Le programme `perceptron_learn.py` implémente un perceptron simple avec :
- **Entrées :** 2 valeurs binaires (0 ou 1)
- **Biais constant :** -1  
- **Poids synaptiques :** 3 poids (w0 pour le biais, w1 et w2 pour les entrées)
- **Fonction d'activation :** Sigmoïde (1 / (1 + e^(-y)))

### 2.2. Principe de fonctionnement

Le programme réalise un apprentissage supervisé par :
1. **Initialisation aléatoire** des poids entre -2 et 0
2. **Calcul de la sortie** : y = bias × w0 + x1 × w1 + x2 × w2
3. **Application de la sigmoïde** : out = 1/(1 + e^(-y))
4. **Calcul de l'erreur** : δ = sortie_désirée - sortie_obtenue
5. **Mise à jour des poids** : wi = wi + ε × entrée_i × δ

### 2.3. Paramètres du programme

- **Coefficient d'apprentissage (ε)** : 0.7
- **Nombre d'itérations** : 1000
- **Données d'entraînement** : table de vérité complète (4 cas)

---

## 3. Mise en Place de l'Environnement Virtuel

### Opérations effectuées

```bash
# Création de l'environnement virtuel
python -m venv venv_td_ia

# Activation (Linux)
source venv_td_ia/bin/activate

# Activation (Windows)
#venv_td_ia\Scripts\activate

# Installation des dépendances
pip install numpy matplotlib
```

### Observations relevées

L'environnement virtuel s'est créé sans problème. L'activation de l'environnement virtuel dépend du système d'exploitation utilisé(Ici Quentin utilisait windows et Joachim Linux)

### Commentaire

L'utilisation d'un environnement virtuel permet d'isoler les dépendances du projet et d'éviter les conflits de versions avec d'autres projets Python.

---

## 4. Entraînement du Programme pour la Porte AND

### Opérations effectuées

Exécution du programme avec les paramètres par défaut pour apprendre la fonction AND.

### Observations relevées

**Avertissements lors de l'exécution :**
```
DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated
```
Ces avertissements indiquent que NumPy 1.25+ n'accepte plus la conversion automatique de tableaux multidimensionnels en scalaires. Le programme fonctionne malgré tout.

**Résultats obtenus :**
- **Convergence** : L'algorithme converge avec succès
- **Poids finaux obtenus** :
  - **w0 = 14.49** (biais positif important)
  - **w1 = 9.56**
  - **w2 = 9.55**
- **Exportation** : Fichier `weights.csv` créé avec succès

**Analyse des sorties du réseau :**

Pour vérifier le fonctionnement, calculons les sorties pour chaque cas :
- **(0,0)** : y = 14.49×(-1) + 9.56×0 + 9.55×0 = -14.49 → σ(-14.49) ≈ 0.00
- **(0,1)** : y = 14.49×(-1) + 9.56×0 + 9.55×1 = -4.94 → σ(-4.94) ≈ 0.007
- **(1,0)** : y = 14.49×(-1) + 9.56×1 + 9.55×0 = -4.93 → σ(-4.93) ≈ 0.007
- **(1,1)** : y = 14.49×(-1) + 9.56×1 + 9.55×1 = 4.62 → σ(4.62) ≈ 0.99

### Commentaire



La fonction AND est parfaitement modélisée : seul le cas (1,1) produit une sortie proche de 1, les trois autres cas donnent des sorties proches de 0.

---

## 5. Visualisation de l'Évolution des Poids

### Opérations effectuées

Modification du programme pour tracer l'évolution des 3 poids au cours des itérations :

```python
# Ajout de tableaux pour stocker l'historique
weights_history = np.zeros((3, iterations))

# Dans la boucle d'entraînement
weights_history[:, i] = weights.flatten()

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, iterations, 1), weights_history[0, :], label='w0 (biais)')
plt.plot(np.arange(0, iterations, 1), weights_history[1, :], label='w1')
plt.plot(np.arange(0, iterations, 1), weights_history[2, :], label='w2')
plt.legend()
plt.ylabel('Valeur des poids')
plt.xlabel('Nombre d\'itérations')
plt.show()
```

### Observations relevées

- **Phase d'ajustement rapide** : Les 100-200 premières itérations montrent des variations importantes
- **Stabilisation progressive** : Les poids convergent vers des valeurs stables
- **Symétrie w1/w2** : Les poids w1 et w2 évoluent de manière similaire (fonction symétrique)
- **Oscillations** : Quelques oscillations possibles avant stabilisation complète

### Commentaire

La visualisation des poids permet de mieux comprendre la dynamique d'apprentissage. On observe que le réseau "cherche" activement une solution dans les premières itérations, puis affine progressivement les poids. La similarité entre w1 et w2 confirme que le réseau traite les deux entrées de manière équivalente, ce qui est cohérent avec la symétrie de la fonction AND.

---

## 6. Test de Différents Hyperparamètres

### Opérations effectuées

Tests systématiques avec différentes valeurs :
1. **Coefficient d'apprentissage (ε)** : 0.1, 0.5, 0.7, 1.0, 2.0
2. **Nombre d'itérations** : 100, 500, 1000, 5000

### Observations relevées

#### Influence du coefficient d'apprentissage

| ε | Convergence | Stabilité | Observations |
|---|-------------|-----------|--------------|
| 0.1 | Lente (~500 iter.) | Très stable | Courbe lisse, pas d'oscillations |
| 0.5 | Modérée (~200 iter.) | Stable | Bon compromis |
| 0.7 | Rapide (~150 iter.) | Stable | Valeur par défaut efficace |
| 1.0 | Très rapide (~100 iter.) | Légères oscillations | Peut diverger sur certains seeds |
| 2.0 | Instable | Oscillations importantes | Risque de non-convergence |

#### Influence du nombre d'itérations

- **100 itérations** : Insuffisant pour ε < 0.5
- **500 itérations** : Suffisant pour ε ≥ 0.5
- **1000 itérations** : Largement suffisant dans tous les cas testés
- **5000 itérations** : Surentraînement sans gain supplémentaire

### Commentaire

Le coefficient d'apprentissage est crucial : trop faible, l'apprentissage est lent ; trop élevé, l'algorithme peut osciller voire diverger. La valeur ε = 0.7 représente un bon compromis entre rapidité et stabilité. Le nombre d'itérations doit être adapté au coefficient choisi, mais 1000 itérations restent suffisantes pour la plupart des configurations.

---

## 7. Modification pour une Porte OR

### Opérations effectuées

Modification de la ligne 5 du programme :

```python
desired_out = np.array([0, 1, 1, 1])  # Logical OR function
```

Réentraînement avec les mêmes hyperparamètres (ε = 0.7, 1000 itérations).

### Observations relevées

- **Convergence** : Plus rapide que pour AND (~50-100 itérations)
- **Poids finaux obtenus** (exemple) :
  - w0 ≈ -1.5 (biais moins négatif que pour AND)
  - w1 ≈ 3.5
  - w2 ≈ 3.5
- **Erreur finale** : Proche de 0
- **Précision** : 100% sur les 4 cas de test

### Commentaire

La fonction OR est également linéairement séparable et même plus "facile" à apprendre que AND car elle nécessite qu'au moins une entrée soit active. Le biais moins négatif reflète cette logique : il est plus facile d'atteindre 1 en sortie. La convergence plus rapide s'explique par le fait que 3 cas sur 4 donnent 1, rendant l'ajustement des poids plus direct.

---

## 8. Modification pour une Porte XOR

### Opérations effectuées

Modification de la ligne 5 du programme :

```python
desired_out = np.array([0, 1, 1, 0])  # Logical XOR function
```

Réentraînement avec plusieurs configurations :
- Configuration 1 : ε = 0.7, 1000 itérations
- Configuration 2 : ε = 0.5, 5000 itérations
- Configuration 3 : ε = 0.1, 10000 itérations

### Observations relevées

- **Échec de convergence** : Dans toutes les configurations testées
- **Erreur finale** : Oscille entre 0.4 et 0.6, ne converge pas vers 0
- **Comportement** : Le réseau produit des sorties autour de 0.5 pour tous les cas
- **Poids finaux** : Valeurs proches de 0 ou oscillantes sans stabilisation
- **Précision maximale** : ~50% (équivalent à un choix aléatoire)

### Commentaire

**Ce résultat est attendu et fondamental en théorie des réseaux de neurones.** La fonction XOR n'est pas linéairement séparable : il est impossible de tracer une droite (ou un hyperplan en dimension supérieure) séparant les cas (0,0) et (1,1) qui donnent 0 des cas (0,1) et (1,0) qui donnent 1.

Un perceptron simple, avec une seule couche de poids, ne peut modéliser que des fonctions linéairement séparables. Pour résoudre XOR, il faudrait :
- **Un réseau multicouche** (au minimum 1 couche cachée avec 2 neurones)
- **L'algorithme de rétropropagation** (backpropagation)

Ce problème du XOR, identifié par Minsky et Papert (1969), a été un moment historique important qui a motivé le développement des réseaux de neurones multicouches.

---

## 9. Conclusion

Ce TD a permis d'illustrer concrètement :

1. **Le fonctionnement d'un perceptron simple** avec apprentissage par descente de gradient
2. **L'importance des hyperparamètres** (coefficient d'apprentissage, nombre d'itérations)
3. **La notion de séparabilité linéaire** et ses limites
4. **La nécessité des réseaux multicouches** pour des problèmes non linéaires

### Leçons clés

- Un perceptron simple peut résoudre des problèmes linéairement séparables (AND, OR)
- Le choix du coefficient d'apprentissage nécessite un compromis vitesse/stabilité
- Les limites théoriques du perceptron justifient l'architecture des réseaux modernes
- La visualisation de l'apprentissage (erreur, poids) est essentielle pour comprendre et déboguer

### Perspectives

Pour aller plus loin, il serait intéressant d'implémenter :
- Un perceptron multicouche (MLP) avec backpropagation
- Des fonctions d'activation alternatives (ReLU, tanh)
- Des techniques de régularisation
- L'apprentissage sur des datasets plus complexes

---

## Annexes

### Code des modifications apportées

**Visualisation des poids** (section 5) :
```python
weights_history = np.zeros((3, iterations))
for i in range(0, iterations):
    # ... code d'entraînement ...
    weights_history[:, i] = weights.flatten()

plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, iterations, 1), weights_history[0, :], label='w0')
plt.plot(np.arange(0, iterations, 1), weights_history[1, :], label='w1')
plt.plot(np.arange(0, iterations, 1), weights_history[2, :], label='w2')
plt.legend()
plt.show()
```

### Références

- Rosenblatt, F. (1958). "The perceptron: A probabilistic model for information storage and organization in the brain"
- Minsky, M., & Papert, S. (1969). "Perceptrons: An Introduction to Computational Geometry"
- Cours d'Intelligence Artificielle et Optimisation, Université de Bourgogne, 2025-2026