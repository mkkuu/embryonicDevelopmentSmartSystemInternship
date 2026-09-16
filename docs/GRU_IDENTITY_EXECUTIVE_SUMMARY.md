# GRU vs Identity Dynamics — Executive Summary

Full analysis: `docs/GRU_IDENTITY_ANALYSIS.md`. Source data: `Results/evaluation/gru_identity_threshold_analysis/`.

### Méthode

Comparaison du GRU (formulation classification, seed=0, checkpoint gelé) et d'Identity Dynamics (baseline sans dynamique temporelle) sur le split Val (106 vidéos, 43 339 fenêtres), un support strictement commun aux deux modèles. Correction méthodologique appliquée : les seuls résultats historiques persistés pour Identity Dynamics dans ce projet avaient été calculés sur le split Test, et non Val — ils ont donc été re-dérivés sur Val (ajustement déterministe, sans ré-entraînement) pour permettre une comparaison valide. Le GRU a été chargé depuis son checkpoint gelé, jamais ré-entraîné. Le split Test n'a jamais été utilisé dans cette analyse.

### Résultats

| Modèle | AUROC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|
| Identity Dynamics | 0,7501 | 0,3250 | 0,1033 | 0,0104 |
| GRU | 0,7638 | 0,3565 | 0,1015 | 0,0224 |

Trois points de fonctionnement du GRU illustrent le compromis sensibilité/faux positifs :

| Régime | Seuil | Recall | Precision | FPR |
|---|---:|---:|---:|---:|
| Haute sensibilité | 0,059 | 90,4 % | 19,5 % | 56,8 % |
| Équilibré (max F1) | 0,184 | 55,8 % | 28,7 % | 21,2 % |
| Haute précision | 0,861 | 0,4 % | 81,5 % | <0,1 % |

Le split Val contient 816 événements de transition réels (définis à partir de la vérité terrain) : le GRU en détecte 85,3 % au seuil équilibré et 99,0 % au seuil haute sensibilité. Les détections ne sont, en médiane, pas en retard par rapport au début de l'événement — mais la durée moyenne d'un événement (~7 fenêtres, un artefact du fenêtrage par recouvrement) rend la référence temporelle exacte (début vs milieu de l'événement) ambiguë ; cette ambiguïté est documentée, pas résolue.

### Analyse

Le GRU surpasse Identity Dynamics sur toutes les métriques de ranking indépendantes du seuil (AUROC, PR-AUC), avec un gain modeste mais mesurable. Aucun seuil unique n'offre simultanément un rappel et une précision élevés : le compromis est réel et quantifié, pas seulement qualitatif. Le régime haute sensibilité (90,4 % de rappel) s'accompagne d'un taux de faux positifs de 56,8 %, ce qui exclut son usage comme déclencheur autonome sans étape de confirmation. Ces écarts entre modèles sont des estimations ponctuelles : aucun test statistique de significativité n'a été réalisé, et un seul seed du GRU a été évalué.

Les résultats suggèrent que le GRU peut constituer un signal de détection ou de localisation temporelle pour le futur système RAG, à condition de ne pas l'utiliser comme classifieur final autonome. Ce signal s'intégrerait comme une couche complémentaire — jamais comme source de vérité — en amont du Reporting API et du Semi-HMM gelé, qui restent les sources de vérité pour toute sortie probabiliste.

### Limites

Déséquilibre de classes (13,24 % de positifs) ; un seul seed GRU évalué (2 autres disponibles mais non exécutés) ; calibration imparfaite du score GRU (ECE=0,0224, à ne pas interpréter comme une probabilité calibrée) ; ambiguïté de la référence temporelle (début vs milieu d'événement) ; marge de recherche des détections bornée à ±15 fenêtres (14,5 % des détections atteignent cette limite) ; absence de test de significativité statistique sur l'écart GRU/Identity Dynamics ; un résultat historique antérieur (non revérifié ici) montre que l'avantage apparent de cette formulation du GRU sur le split Test s'inverse sous un test de mélange des frames (frame-shuffle), laissant ouverte la question de la part réelle de raisonnement temporel.

### Conclusion

Le GRU apporte un gain de ranking mesurable mais modeste par rapport à Identity Dynamics, sans constituer un classifieur final performant à un seuil unique. Il est défendable comme signal d'alerte ou de localisation temporelle pour un futur système RAG, à condition explicite qu'il ne soit jamais traité comme une vérité biologique ni comme un classifieur autonome — conclusion cohérente avec l'architecture produit déjà établie de ce projet, où le Reporting API reste l'unique source de vérité pour les valeurs numériques présentées à l'utilisateur.
