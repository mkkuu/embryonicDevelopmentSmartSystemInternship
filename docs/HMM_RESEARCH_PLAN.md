# HMM / Semi-HMM — Étude scientifique et technique

Écrit 2026-08-19, avant tout code. Compagnon de `RESEARCH_BLUEPRINT.md` (programme scientifique), `HANDOFF.md` (formulation mathématique), `PROJECT_STATE.md` (état E1/GRU). Ne remplace aucun de ces documents.

**Règle absolue, rappelée et respectée dans tout ce document** : E1 (base_rate/identity_dynamics/persistence/linear_ssm) et GRU (STEP 6A/6B/6C) sont des **baselines expérimentales verrouillées**, jamais retouchées, jamais réécrites comme "corrigées" par ce qui suit. Le HMM est une **nouvelle branche**, ajoutée par composition — aucun fichier de `Training/` original ni aucun artefact `Results/evaluation/e1_*`/`e1_gru_*` n'est modifié par ce plan.

---

## 0. Cadrage — ce bifurcation n'est pas nouvelle, elle était déjà prévue

Point important découvert pendant l'audit (§1) : le HMM **n'est pas une idée hors programme**. `HANDOFF.md` §4.4 point 6 le nomme explicitement comme "branche séparée" de la chaîne de simplification du modèle génératif : *"discard continuous x entirely, model only r(t) as a semi-Markov process with duration-dependent hazards — the HSMM"*. `RESEARCH_BLUEPRINT.md` Part II reprend ce point 6 à l'identique, et surtout **Part VIII liste "plain HMM (isolates discreteness from duration-awareness)" comme un baseline jamais construit** de l'échelle E1, à côté de "classifier + post-hoc temporal smoothing" — également jamais construit, et explicitement marqué *"the cheapest real competitor — must be beaten, not skipped"* dans Part VII.

Cela crée une distinction centrale qu'il faut trancher avant toute implémentation, parce que E1 a été mesuré **négatif** (persistence/linear_ssm perdent contre identity_dynamics) et GRU aussi (voir `PROJECT_STATE.md` §3b/§3j-k), et `RESEARCH_BLUEPRINT.md` Part III a une règle de décision explicite : *"Clear negative (post-sanity-check) → do not build RQ2/RQ3; execute the pivot directions."* RQ2, c'est le modèle de régime latent (H2) — **si le HMM qu'on construit est une instance de RQ2, la règle du programme dit officiellement de ne pas le construire maintenant, sauf décision explicite de réouverture.**

Il y a donc **deux HMM possibles**, avec un statut différent vis-à-vis de cette règle :

| Formulation | État caché | Statut vis-à-vis de la règle E1-négatif | Question testée |
|---|---|---|---|
| **(a) HMM-classifieur contraint** | Phase biologique observée (label existant, pas latent) | **Non concerné** — ce n'est pas RQ2 (régime latent), c'est le baseline "classifier + smoothing" jamais construit, une question de décodage structuré, pas de dynamique latente | Le décodage contraint par transitions (Viterbi/forward-backward) améliore-t-il l'identification de phase/transition par rapport à une classification fenêtre-par-fenêtre indépendante ? |
| **(b) HMM-régime latent (= E2/H2)** | Régime non observé (normal/arrêté/dégénérescent, K≈3) | **Concerné** — c'est RQ2 tel que défini dans `RESEARCH_BLUEPRINT.md` Part III, explicitement gaté par RQ1 | Un régime discret latent explique-t-il un comportement futur qu'un état continu n'explique pas ? |

**Recommandation retenue pour ce plan** (justifiée en détail §2, §13) : commencer par **(a)**, qui n'est pas soumis au gate RQ1-négatif et comble une lacune documentée du programme existant. **(b)** reste possible mais nécessite une décision utilisateur explicite de rouvrir RQ2 malgré le résultat négatif de RQ1 — ce plan ne la prend pas silencieusement.

---

## 1. Étape 1 — Audit de l'existant

| Expérience | Modèle | Données | Métriques | Artefacts | Statut |
|---|---|---|---|---|---|
| E1 classification | base_rate, identity_dynamics, persistence, linear_ssm | Embeddings resnet18_balanced, 512D, Train 492/Val 106/Test 106 vidéos | Accuracy, AUROC, precision/recall/F1, bootstrap patient-level (n=1000), Holm-Bonferroni sur 3 comparaisons pré-enregistrées | `Results/evaluation/e1_full_analysis/` (remote uniquement) | **Verrouillé, ne pas toucher** |
| E1 forecast/imputation | linear_ssm | idem | MSE k-step forecast, MSE imputation, vs baseline naïve | `Results/evaluation/linear_ssm_dynamics_report.json` | **Verrouillé** |
| E1 frame-shuffle | les 4 modèles E1 | idem, ordre des fenêtres mélangé | Δ AUROC réel vs shuffled | `Results/evaluation/e1_frame_shuffle/` | **Verrouillé** |
| GRU STEP 6A/B/C | GRUDynamicsModel (`residual`, `direct`), 3 seeds | idem | Idem E1 + comparaison vs identity_dynamics/linear_ssm, frame-shuffle | `Results/evaluation/e1_gru_step6{a,b,c}_*/` | **Verrouillé** |
| Tests unitaires | 4 modèles E1 + GRU | Synthétique | — | `Tests/evaluation/` | 79/79, local + remote |

**Données/pipeline reconstruits (lecture de `DataSet.py`, `preProcess.py`, `trajectory.py`, `model.py`, `metrics.py`)** :

- **Phases** (`Training/DataSet.py:151-167`) : `chronological_phases = [tPB2, tPNa, tPNf, t2, t3, t4, t5, t6, t7, t8, t9+, tM, tSB, tB, tEB]` — 15 phases, ordre biologique fixe. `tHB` est explicitement exclue (`DataSet.py:123`). C'est un vocabulaire **existant dans le repo**, pas une invention — utilisable tel quel pour §2 option B.
- **Fenêtres** : glissantes, `window_size=8` (resnet18), `stride=1`, par vidéo, triées par identifiant de frame. Une fenêtre est rejetée (sauf `multiple_phases=True`) si elle couvre >2 phases, ou 2 phases non adjacentes dans `chronological_phases`.
- **`consistency_flag`** : `0` si `first_frame_phase == last_frame_phase` (fenêtre stable), `1` sinon (fenêtre de transition) — la cible actuelle de tout le ladder E1/GRU. **Pas** le label de phase lui-même.
- **Labels disponibles par fenêtre** : `first_frame_phase`, `last_frame_phase` (indices dans `chronological_phases`) — déjà propagés jusqu'à `evaluation.trajectory.Trajectory` (`first_frame_phase: torch.Tensor (T,)`, `last_frame_phase: torch.Tensor (T,)`). **Ce sont des labels de phase gratuits, déjà dans le pipeline, pour chaque fenêtre de chaque trajectoire** — voir §2.
- **Temporalité** : `Data/embryo_dataset_time_elapsed/` existe dans les données brutes (temps réel inter-frame) mais **n'est jamais joint** aux CSV d'entraînement (`preProcess.py::process_annotations` ne le référence jamais — confirmé par lecture du fichier). Sans ce join, aucune durée n'est exprimable en heures réelles — seulement en nombre de fenêtres (unité arbitraire, dépendante de `stride`). C'est un vrai manque, pas une supposition — voir Phase 1 du plan (§12).
- **Train/Val/Test** : split par vidéo (`preProcess.py::create_splits`, `train_test_split` à seed 42, 70/15/15), jamais par fenêtre — pas de fuite patient. 492/106/106 vidéos.
- **`Trajectory`/`Model`/`metrics` (évaluation)** : interface `fit(train, val)/predict(trajectories)→List[TrajectoryPrediction]/save/load`, déjà découplée du modèle concret (registre par nom). `bootstrap_trajectory_metrics`/`paired_bootstrap_comparison`/`holm_bonferroni` déjà génériques, indépendants du modèle. `shuffle_trajectory` déjà générique. **Tout ceci est directement réutilisable pour le HMM sans aucune modification** — un point important pour l'estimation d'effort (§12).

Aucun artefact existant n'a été lu au-delà de sa description dans `PROJECT_STATE.md` — aucune écriture, aucune commande destructive n'a été exécutée pendant cet audit.

---

## 2. Étape 2 — Définir les états

| Option | Données nécessaires | Supervision | Interprétabilité | Compatible reporting | Risque de circularité | Risque de leakage |
|---|---|---|---|---|---|---|
| **A. Phases biologiques** (`chronological_phases`, 15 classes) | Déjà présentes (`first_frame_phase`/`last_frame_phase`) | Supervisée (labels d'annotation existants) | Maximale — vocabulaire clinique direct | Idéal — "Phase B" est déjà le nom clinique | **Faible-moyen** — voir note ci-dessous | Faible : labels déjà utilisés pour construire `consistency_flag`, même source de vérité que E1/GRU |
| **B. Classes déjà disponibles** (= identique à A, `first_frame_phase`/`last_frame_phase` du pipeline) | Déjà présentes | Supervisée | Maximale | Idéal | Idem A | Idem A |
| **C. Phases temporelles** (bucketing par temps écoulé, ex. quartiles d'heures) | `embryo_dataset_time_elapsed/`, **actuellement non joint** | Semi-supervisée / dérivée | Faible — pas de sens clinique direct | Mauvais — un "état" numérique n'est pas explicable au médecin | Faible | Faible, mais ajoute une dépendance de données non encore construite |
| **D. États latents non supervisés** (clustering sur les embeddings, K appris) | Déjà présentes (embeddings) | Non supervisée | Faible à moyenne — nécessite un post-hoc mapping vers le vocabulaire clinique pour être utile | Mauvais tant que le mapping n'est pas fait | **Élevé si K et le mapping sont choisis après avoir regardé les résultats** | Faible sur les données, mais fuite méthodologique si K est choisi post-hoc sans se pré-enregistrer |
| **E. Régime latent (E2/H2)** (normal/arrêté/dégénérescent, K≈3) | Déjà présentes, mais **aucun exemple annoté d'arrêt/dégénérescence n'est confirmé disponible** (`RESEARCH_BLUEPRINT.md` Part VI : "Anomalous/regime data scarcity — High probability... no curation resource confirmed") | Non supervisée ou faiblement supervisée | Élevée si ça marche, nulle si le régime ne se sépare pas des phases | Bon, si validé | Moyen | Moyen — sans exemples confirmés d'anomalie, le régime risque de recoller exactement aux phases (redondant avec A), rendant la "découverte" triviale |

**Recommandation** : **B ≡ A** comme état caché du HMM de base (Étape 3/4). Justification :
- Gratuit : déjà dans `Trajectory.first_frame_phase`/`last_frame_phase`, zéro nouveau travail d'étiquetage.
- Même source de supervision qu'E1/GRU → comparaison loyale, pas de changement de protocole caché.
- Directement interprétable dans le reporting (Étape 7) — "Phase B" est littéralement le nom clinique.
- **Note sur le risque de circularité (A/B)** : les phases sont *par construction* ordonnées dans le temps (elles sont définies par l'annotateur humain à partir du temps écoulé lui-même). Un HMM contraint qui "découvre" que l'ordre chronologique est respecté ne découvre donc rien de nouveau — ce n'est pas circulaire au sens statistique (pas de fuite train/test), mais c'est **tautologique** au sens scientifique si on prétend que la contrainte d'ordre est "apprise". Il faut présenter la contrainte d'ordre comme un **choix de modélisation justifié par la littérature clinique** (les phases sont définies chronologiquement), pas comme un résultat empirique. Le résultat empirique intéressant n'est pas "l'ordre est respecté", c'est : **le décodage contraint améliore-t-il l'identification de phase à partir des images**, en particulier sur les fenêtres ambiguës (`consistency_flag=1`) — une question non triviale, mesurable, indépendante de la tautologie d'ordre.
- **C est explicitement rejetée pour l'instant** — dépend d'un join de données non fait, et un état défini par le temps lui-même serait circulaire pour toute tâche qui prétendrait ensuite *estimer* le temps/la durée (Étape 5) : on ne peut pas définir l'état par la durée et ensuite "découvrir" une durée par état, c'est la définition même de la métrique retournée sur elle-même. C ne sera reconsidérée que si A/B échoue et qu'une réponse temporelle plus fine est spécifiquement motivée.
- **D est explicitement déferrée** — c'est en réalité RQ4/H4 (`RESEARCH_BLUEPRINT.md` Part III, "aspirational, deferred"), déjà positionnée dans le programme comme dépendante d'un état biologique validé au préalable. La faire maintenant serait sauter une étape déjà justifiée comme prématurée.
- **E (régime latent, = E2/H2 réel) n'est pas rejetée mais n'est pas le point d'entrée** — voir §0. Elle reste l'extension naturelle si A/B ne suffit pas à expliquer des trajectoires atypiques, mais nécessite (i) une décision explicite de réouverture de RQ2 malgré le résultat négatif de RQ1, (ii) une vraie source d'exemples anormaux, actuellement non confirmée.

---

## 3. Étape 3 — HMM de base (état = phase, §2 option A/B)

**Définitions**

- `O_t ∈ ℝ^512` — l'embedding resnet18_balanced de la fenêtre `t` (déjà caché, `Trajectory.embeddings[t]`).
- `S_t ∈ {1,...,15}` (ou moins, selon les phases présentes dans le split) — la phase, alignée sur `last_frame_phase[t]` par défaut (voir note d'alignement ci-dessous).
- `π_i = P(S_1 = i)` — distribution initiale, estimée par fréquence empirique des premières fenêtres de chaque trajectoire d'entraînement.
- `A_{ij} = P(S_t = j | S_{t-1} = i)` — matrice de transition, estimée par comptage de bigrammes `(S_{t-1}, S_t)` sur les trajectoires d'entraînement (Baum-Welch non nécessaire ici puisque `S_t` est **observé** à l'entraînement — apprentissage **supervisé**, pas EM).
- `P(O_t | S_t = i)` — modèle d'émission. Recommandation : **hybride discriminatif**, pas un mélange de gaussiennes en 512D (trop de paramètres pour ~500 trajectoires). On réutilise directement l'idée déjà en place dans `identity_dynamics.py` (logistic regression sur l'embedding), étendue au multi-classe (15 phases) : entraîner `P(S_t=i | O_t)` par régression logistique multinomiale, puis convertir en pseudo-vraisemblance par la règle de Bayes :

  ```
  P(O_t | S_t=i) ∝ P(S_t=i | O_t) / P(S_t=i)
  ```

  (le "hybride HMM/réseau discriminatif" — technique standard depuis Bourlard & Morgan 1994 en reconnaissance de la parole, directement applicable ici, et cohérente avec la discipline déjà établie du projet — JSON-only, pas de nouveau réseau profond, réutilise `sklearn.linear_model.LogisticRegression` déjà utilisé partout dans `evaluation/models/`).

**Apprentissage** : `π`, `A` par comptage supervisé (fermé, pas d'itération). Émission par régression logistique multinomiale standard (`sklearn`, déjà la brique utilisée par `identity_dynamics`/`persistence`/`linear_ssm`).

**Inférence — trois algorithmes distincts, à ne jamais confondre dans le reporting (Étape 7)** :

1. **Forward** : `α_t(i) = P(O_{1:t}, S_t=i)`, récursif : `α_t(j) = [Σ_i α_{t-1}(i) A_{ij}] · P(O_t|S_t=j)`. Donne `P(O_{1:T}) = Σ_i α_T(i)` — la vraisemblance totale de la séquence sous le modèle (utile pour la log-vraisemblance/calibration, Étape 10).
2. **Backward** : `β_t(i) = P(O_{t+1:T} | S_t=i)`, récursif en arrière.
3. **Forward-backward (lissage/smoothing)** : `γ_t(i) = P(S_t=i | O_{1:T}) = α_t(i)β_t(i) / P(O_{1:T})` — la probabilité **a posteriori**, utilise **toute** la séquence (passé ET futur). Non causal.
4. **Filtrage (forward seul)** : `P(S_t=i | O_{1:t}) = α_t(i) / Σ_j α_t(j)` — n'utilise que le passé. **Causal.**
5. **Viterbi** : `δ_t(i) = max_{s_{1:t-1}} P(s_{1:t-1}, S_t=i, O_{1:t})`, avec backtracking → la séquence d'états `argmax_{s_{1:T}} P(s_{1:T} | O_{1:T})` la plus probable **globalement** (un seul chemin, pas une distribution). Non causal (utilise `O_{1:T}` en entier avant de choisir le meilleur chemin).

**Point méthodologique critique, à répéter dans le reporting (Étape 7)** : pour un usage clinique en temps réel (l'embryon est encore en cours de développement, on n'a pas encore vu la fin de la vidéo), **seul le filtrage (4) est valide** — la même discipline de causalité déjà appliquée au GRU (`test_gru_is_causal`, `test_gru_no_future_information_in_forecast`, voir `Tests/evaluation/models/test_gru.py`) s'applique ici et doit être testée de la même façon (`test_hmm_filtering_is_causal`). Le lissage (3) et Viterbi (5) sont valides seulement pour une analyse rétrospective complète d'une trajectoire déjà terminée (ex. rapport final, recherche) — les mélanger dans un même reporting sans le dire serait une erreur de méthode, pas un détail.

**Objectif de cette étape** : déterminer si un décodage structuré (contrainte de transition + lissage) améliore l'identification de phase et de transition par rapport à `identity_dynamics` (fenêtre-par-fenêtre, indépendante) — comparaison directe et loyale, même embeddings, même split, même infrastructure de bootstrap.

**Note d'alignement fenêtre→état** : une fenêtre peut couvrir 2 phases adjacentes (`consistency_flag=1`). Recommandation : `S_t := last_frame_phase[t]` (label aligné à droite, cohérent avec le filtrage causal — "l'état tel qu'il est à la fin de la fenêtre la plus récente"). Documenter la fraction de fenêtres ambiguës par paire de phases (déjà calculable via `_print_transition_matrix` de `DataSet.py`, à reproduire côté `evaluation/`) et suivre spécifiquement la performance de décodage sur ce sous-ensemble (= la population `consistency_flag=1`, déjà la métrique "recall Transition" suivie depuis E1).

---

## 4. Étape 4 — HMM contraint

**Contraintes étudiées, dérivées de ce qui existe déjà dans le repo/`HANDOFF.md` §4.6 (pas inventées)** :

- **Ordre chronologique / transitions interdites** : `A_{ij} = 0` si `j` n'est ni `i` ni le successeur immédiat de `i` dans `chronological_phases` — exactement la règle déjà codée dans `DataSet.py::_create_sequences` (adjacence de phase) et déjà justifiée par `HANDOFF.md` §4.6 ("Irreversibility — Hard, sparsity on Q").
- **Pas de retour en arrière** : sous-cas du point précédent (`A_{ij}=0` pour `j<i`) — cohérent avec `HANDOFF.md` §4.6.
- **Maintien dans une phase** : `A_{ii}` libre (pas contraint à 0), seule contrainte est sur les sauts, pas sur l'auto-transition.
- **État initial** : `π` restreint aux phases effectivement observées en première fenêtre du split (déjà garanti empiriquement, pas besoin de forcer).
- **État terminal** : optionnel — si on veut un HMM "left-to-right"/Bakis strict (utile pour Étape 5/7, voir ci-dessous), on peut interdire toute transition sortante depuis la dernière phase observée sauf vers elle-même.

**Comparaison HMM libre vs HMM contraint — quantitative** :
- Log-vraisemblance totale sur Val/Test (`P(O_{1:T})` du forward), comparée entre les deux variantes — un HMM contraint a *structurellement* une vraisemblance ≤ au libre (moins de paramètres/chemins possibles) ; l'intérêt n'est pas de maximiser la vraisemblance mais de vérifier que la perte de vraisemblance est négligeable (preuve que la contrainte n'est pas fausse) tout en gagnant en robustesse de décodage.
- Accuracy/recall de décodage de phase et de transition (mêmes métriques que E1, réutilisation directe de `evaluation/metrics.py`), avec IC bootstrap patient-level, comparant libre vs contraint.
- Fraction de trajectoires Test dont le décodage Viterbi **libre** viole déjà la contrainte chronologique (mesure directe de si la contrainte "mord" ou est déjà satisfaite naturellement par les données — répond à la note de circularité §2).

**Lien direct avec Étape 5/7** : un HMM contraint "left-to-right" (pas de retour en arrière, un seul passage par état) transforme le problème "quelle est la probabilité de la trajectoire A→B→C→D" en un problème **tractable et bien défini** — parce que le chemin d'état, sous cette contrainte, se réduit à une séquence de *durées de séjour* dans chaque phase visitée. C'est exactement la structure d'un modèle à durée explicite (Semi-HMM/segmental), pas une coïncidence — voir §7.

---

## 5. Étape 5 — Semi-HMM : critère de décision, pas un choix a priori

**Ce qui doit être vrai dans les données pour justifier un Semi-HMM** :
1. La durée réelle par phase (`D | S=k`) doit être **estimable** — nécessite le join `embryo_dataset_time_elapsed` (§1, actuellement absent). Sans ce join, "durée" ne serait exprimable qu'en nombre de fenêtres (`stride`-dépendant, non clinique) — **insuffisant pour l'exemple du prompt ("Durée estimée de A : 8.4h")**, qui exige des heures réelles.
2. La distribution de durée par phase doit être **non-géométrique** de façon significative. Un HMM standard implique implicitement `D|S=k ~ Géométrique(1-A_{kk})` (durée de séjour = nombre d'essais avant le premier "échec" d'auto-transition) — une hypothèse très restrictive (masse concentrée sur les courtes durées, décroissance monotone). Si la durée réelle par phase (une fois §1 joint) est mieux ajustée par une Gamma/log-normale/Weibull à mode non nul, c'est une preuve directe que le Semi-HMM apporte quelque chose que le HMM standard ne peut pas représenter, indépendamment de toute métrique de classification.
3. Test concret, peu coûteux, à faire **avant** d'écrire le moindre code de Semi-HMM : sur Train (une fois §1 fait), pour chaque phase, comparer l'ajustement (AIC/BIC, ou simplement QQ-plot) d'une géométrique vs une Gamma/log-normale sur les durées réelles observées. **Ne pas passer au Semi-HMM si le test 2 échoue** (géométrique déjà adéquate) — ce serait de la complexité non justifiée, contraire à la discipline déjà établie du projet ("composition over modification", "no premature generalization").

**HMM (durée implicite) vs Semi-HMM (durée explicite)** :

| | HMM standard | Semi-HMM |
|---|---|---|
| `P(D\|S=k)` | Géométrique implicite, non paramétrée séparément | Distribution explicite par état (ex. Gamma(shape_k, scale_k)), estimée sur les durées réelles |
| Algorithme de décodage | Viterbi standard, `O(T·K²)` | Viterbi segmental / EM segmental, `O(T²·K)` ou `O(T·D_max·K)` avec troncature de durée max |
| Sortie "durée estimée de la phase B" | Dérivable indirectement (`E[D]=1/(1-A_{kk})`), mais grossière | Directe, avec intervalle de confiance/quantiles de la distribution ajustée |
| Coût d'implémentation | Faible (comptage + sklearn) | Moyen (décodage segmental à écrire, pas dans une lib standard scikit — `hmmlearn` gère HMM mais pas nativement les durées non-géométriques par état sans extension GMMHMM/estimation manuelle) |

**Conclusion de cette étape (à date, avant tout code)** : ne pas commencer par le Semi-HMM. Faire le test de l'étape 3 ci-dessus dès que §1 (join des temps réels) est fait, **avant** la Phase 6 du plan (§12) — c'est le critère GO/NO-GO explicite pour cette phase, pas une préférence de style.

---

## 6. Étape 6 — Comparaison avec tous les modèles existants

Protocole réutilisé **à l'identique** quand c'est possible : même embeddings (`resnet18_balanced`), même split Train/Val/Test, même bootstrap patient-level (`n_bootstrap=1000`, `rng_seed=0`), même correction Holm-Bonferroni, même infrastructure (`evaluation.metrics`, `evaluation.trajectory`), même frame-shuffle.

| Modèle | Accuracy (transition) | AUROC (transition) | Forecast | Décodage de phase | Robustesse (shuffle) | Interprétabilité |
|---|---:|---:|---:|---:|---|---|
| Base Rate | mesuré (E1) | mesuré (E1) | n/a | n/a | mesuré | Nulle |
| Identity Dynamics | mesuré (E1) | mesuré (E1) | n/a | n/a (pas conçu pour) | mesuré (invariant, par construction) | Faible (boîte noire logistique) |
| Persistence | mesuré (E1) | mesuré (E1) | n/a (pas cette tâche) | n/a | mesuré | Faible |
| Linear SSM | mesuré (E1) | mesuré (E1) | mesuré (E1) | n/a | mesuré | Moyenne (A, b explicites) |
| GRU (forecast/classification) | mesuré (STEP6) | mesuré (STEP6) | n/a (GRU ne fait pas de forecast d'embedding testé ici de la même façon — voir note) | n/a | mesuré | Faible (réseau) |
| **HMM (libre)** | à mesurer | à mesurer | n/a (pas la même tâche que linear_ssm — voir note) | **à mesurer (nouvelle colonne)** | à mesurer (frame-shuffle attendu très destructeur — le HMM dépend structurellement de l'ordre) | **Élevée** — π, A, émissions tous inspectables |
| **HMM (contraint)** | à mesurer | à mesurer | n/a | à mesurer | à mesurer | Élevée |
| **Semi-HMM** | à mesurer (si Phase 6 lancée) | à mesurer | n/a | à mesurer + **durée par phase (nouvelle colonne, absente des modèles précédents)** | à mesurer | Élevée |

**Différences de protocole à expliciter, pas à forcer dans les mêmes colonnes** :
- **"Forecast"** (linear_ssm) mesure une erreur de reconstruction d'embedding (régression, MSE) — un HMM ne prédit pas un embedding futur, il prédit une distribution sur les *états* futurs (`Σ_i γ_t(i) A_{ij}`). Ce ne sont pas comparables numériquement ; le tableau doit dire "n/a — tâche différente", pas afficher un MSE inventé.
- **"Décodage de phase"** est une colonne **nouvelle**, qu'aucun modèle E1/GRU ne rapporte aujourd'hui (ils prédisent uniquement `consistency_flag`, binaire) — c'est un vrai ajout de capacité, pas juste une nouvelle métrique sur la même tâche. Le comparer nécessite d'ajouter cette même sortie aux modèles existants *si on veut une comparaison directe*, ou d'accepter que cette colonne n'a de sens que pour le HMM — à trancher explicitement, pas silencieusement.
- **"Robustesse (shuffle)"** : le HMM et le Semi-HMM devraient être **structurellement très sensibles** au frame-shuffle (contrairement à `identity_dynamics`, invariant par construction) — c'est attendu et souhaitable ici (un modèle séquentiel qui n'est pas affecté par le mélange de l'ordre serait suspect), à l'inverse de la lecture qu'on fait pour GRU/persistence où un score qui *survit* au shuffle est le signal d'alarme. **Ne pas réutiliser mécaniquement la même règle d'interprétation ("survit au shuffle = confondant") sans l'adapter** — pour un modèle dont la structure même *est* la contrainte d'ordre, la bonne question est l'inverse : "le shuffle détruit-il le décodage aussi fortement que prévu, et le HMM sur données mélangées est-il pire qu'`identity_dynamics` (qui, lui, ne dépend pas de l'ordre) ?" C'est un contrôle de validité différent, à définir explicitement avant de lancer l'expérience (voir Phase 4/7 du plan §12).

---

## 7. Étape 7 — Reporting probabiliste : quelles quantités sont quoi

**Ne jamais faire** : `P(A→B→C→D) ≠ A_{AB} · A_{BC} · A_{CD}`. Ce produit naïf ignore (i) l'incertitude sur l'état courant lui-même (on n'observe jamais l'état, seulement l'embedding), (ii) toutes les autres trajectoires possibles compatibles avec les observations, (iii) la durée réelle passée dans chaque état (si Semi-HMM). C'est une erreur classique, explicitement à éviter (le prompt le demandait).

**Ce qui est correctement défini, quantité par quantité** :

| Quantité du reporting | Nom mathématique correct | Formule | Causal (temps réel) ? |
|---|---|---|---|
| "État actuel : Phase B, P(B\|observations)=94%" | **Posterior de filtrage** | `α_t(B) / Σ_i α_t(i)` (utilise `O_{1:t}` seulement) | Oui — c'est la seule quantité valide pour un rapport "en direct" |
| "Transitions possibles : B→C, B→D, B→B" | **Distribution prédictive à un pas** | `P(S_{t+1}=j \| O_{1:t}) = Σ_i [filtrage_t(i)] · A_{ij}` — **pas** juste la ligne `A` de l'état arg-max, qui ignore l'incertitude sur l'état courant | Oui |
| "Transition retenue : B→C, probabilité 82%" | Composante de la distribution prédictive ci-dessus, éventuellement arg-max | idem | Oui |
| "Durée estimée de B" | **Espérance (± IC) de la distribution de durée ajustée pour l'état B** | HMM : `E[D]=1/(1-A_{BB})` (grossier) ; Semi-HMM : moyenne/quantiles de la distribution paramétrique ajustée (§5) | Oui (a priori, avant observation de la sortie de B) ou révisable (a posteriori, une fois `τ` observé — voir `HANDOFF.md` §4.1, `τ(t)` = temps déjà passé) |
| "Trajectoire globale la plus probable : A→B→C→D" | **Chemin Viterbi** (le plus probable globalement) OU, si on utilise la contrainte left-to-right (§4), **le chemin segmental le plus probable** compatible avec un HMM/Semi-HMM contraint | `argmax_{s_{1:T}} P(s_{1:T}\|O_{1:T})` | **Non — rétrospectif seulement**, ne peut être rapporté qu'une fois toute la séquence observée, ou comme "chemin le plus probable *si le développement continue selon le modèle appris*" en partant de l'état filtré courant (une projection, à étiqueter comme telle) |
| "Probabilité / score de cette trajectoire" | **Score de vraisemblance relatif du chemin Viterbi**, `exp(δ_T(best-path)) / P(O_{1:T})` **si** on veut une vraie probabilité (probabilité que ce chemin exact soit le bon, parmi tous les chemins compatibles) — typiquement une petite valeur car il y a de nombreux chemins concurrents. **Alternative recommandée**, plus lisible cliniquement : sous la contrainte left-to-right (§4), la probabilité de la **segmentation** (la séquence ordonnée de phases visitées, indépendamment des durées exactes) est correctement calculable par un forward restreint aux chemins monotones — une vraie probabilité marginale, pas un artefact de normalisation | Non, rétrospectif |
| "Incertitude" | **Entropie de la distribution de filtrage** `H(γ_t) = -Σ_i γ_t(i) log γ_t(i)`, et/ou marge entre les deux états les plus probables | — | Oui |

**Taxonomie à ne jamais confondre (directive explicite du prompt)** :
- **Probabilités** (somment à 1 sur un ensemble bien défini) : `γ_t` (filtrage/lissage), distribution prédictive à un pas, probabilité de segmentation sous contrainte left-to-right.
- **Scores/vraisemblances** (ne somment pas à 1, pas bornés) : `P(O_{1:T})` (vraisemblance totale de la séquence sous le modèle — sert à la log-vraisemblance/calibration, pas à un pourcentage clinique), score Viterbi non normalisé.
- **Estimations ponctuelles avec incertitude** : durée espérée ± IC (Semi-HMM), ou par défaut ± écart-type de la distribution ajustée.
- **Vraisemblance vs posterior** : l'émission `P(O_t|S_t)` (§3) est une vraisemblance, pas une probabilité d'état — ne jamais l'afficher telle quelle comme "confiance dans l'état" sans la faire passer par Bayes (`γ_t`).

**Format du reporting** (repris de l'exemple du prompt, complété avec les libellés exacts ci-dessus) :

```
------------------------------------
PATIENT / TRAJECTOIRE
------------------------------------
État actuel (filtrage causal) :
  Phase B — posterior P(S_t=B | O_1:t) = 0.94

Distribution prédictive à t+1 :
  B → C : 0.82
  B → D : 0.04
  B → B : 0.14

Transition retenue (arg-max de la distribution ci-dessus) :
  B → C, probabilité 0.82

Durée estimée de B (Semi-HMM, si Phase 6 validée) :
  E[D] = 13.2h, IC 90% = [9.8h, 17.1h]
  (HMM standard, à défaut : E[D] = 1/(1-A_BB) fenêtres, à convertir grossièrement)

Incertitude (entropie du posterior courant) :
  H = 0.31 nats (faible incertitude — un seul état domine)

Trajectoire globale la plus probable (RÉTROSPECTIF ou PROJECTION, à étiqueter) :
  A → B → C → D
  Probabilité de cette segmentation (sous contrainte left-to-right) : 0.61
  [Score Viterbi brut, si rapporté séparément : log-vraisemblance = -142.3, NON un pourcentage]
------------------------------------
```

---

## 8. Étape 8 — Format structuré pour le futur RAG

Schéma proposé, avec le typage exact de chaque champ (aligné sur §7, pour que le RAG ne puisse jamais confondre un score et une probabilité) :

```json
{
  "trajectory_id": "Patient_123",
  "model_version": "hmm_constrained_v1",
  "generated_at": "2026-08-19T10:00:00Z",
  "window_index": 214,
  "current_state": {
    "label": "B",
    "type": "posterior_filtering",
    "value": 0.94,
    "causal": true
  },
  "next_state_distribution": [
    {"label": "C", "type": "predictive_probability", "value": 0.82},
    {"label": "D", "type": "predictive_probability", "value": 0.04},
    {"label": "B", "type": "predictive_probability", "value": 0.14}
  ],
  "retained_transition": {
    "from": "B", "to": "C",
    "type": "predictive_probability", "value": 0.82
  },
  "duration_estimate": {
    "state": "B",
    "type": "expected_value_with_interval",
    "unit": "hours",
    "mean": 13.2,
    "interval_90": [9.8, 17.1],
    "source": "semi_hmm_gamma_fit",
    "fallback_if_hmm_only": {"unit": "windows", "mean_geometric": 11.4}
  },
  "trajectory": {
    "type": "viterbi_or_projection",
    "retrospective": false,
    "states": ["A", "B"],
    "projected_states": ["C", "D"],
    "segmentation_probability": {
      "type": "constrained_forward_marginal",
      "value": 0.61
    },
    "raw_viterbi_score": {
      "type": "log_likelihood_unnormalized",
      "value": -142.3
    }
  },
  "uncertainty": {
    "type": "posterior_entropy",
    "unit": "nats",
    "value": 0.31
  },
  "provenance": {
    "embedding_model": "resnet18_balanced",
    "embedding_checksum": "be5460d0...",
    "hmm_config_id": "...",
    "commit": "...",
    "seed": 0
  }
}
```

Règle de conception : **chaque nombre porte son `type`** (`posterior_filtering`, `predictive_probability`, `expected_value_with_interval`, `log_likelihood_unnormalized`, `posterior_entropy`, ...) — pas de champ numérique nu. C'est ce qui permet au RAG de savoir dire *"P(B)=94%"* mais jamais de recalculer ou d'interpoler un score de vraisemblance comme s'il était une probabilité. `provenance` est obligatoire (checksum embedding, config, commit, seed) — cohérent avec la discipline déjà en place pour les checkpoints (`gpu_server_access` memory) et avec Étape 11.

---

## 9. Étape 9 — Application web future

Ce que le HMM doit produire pour rendre l'interface du prompt possible (time-lapse + reporting côte à côte) :
- Une entrée du JSON §8 **par fenêtre déjà vue** (pas seulement la dernière) — pour peupler une timeline scrollable synchronisée avec les images, pas juste un instantané final.
- Le `current_state`/`next_state_distribution` doivent être **causaux** (filtrage, jamais lissage) pour tout point de la timeline avant la dernière fenêtre — sinon l'interface montrerait, pour un point du passé, une confiance qui utilise des images que le médecin n'avait pas encore vues à ce moment-là (fuite visuelle, trompeuse).
- Un flag explicite `retrospective: true/false` sur tout ce qui touche à la "trajectoire globale" (Viterbi complet), pour que l'UI puisse le griser/distinguer visuellement d'une prédiction en direct.
- Rien de plus n'est requis du HMM à ce stade — l'intégration RAG et l'UI elle-même sont hors scope de ce plan (voir `RESEARCH_BLUEPRINT.md` Part V : "Trajectory Analyzer, Knowledge Grounding/RAG — NOT NOW").

---

## 10. Étape 10 — Validation, deux dimensions distinctes

**A. Performance prédictive** (comparable à E1/GRU) : accuracy/recall/F1 sur `consistency_flag` (dérivé du décodage : `1` si l'état décodé change entre fenêtres consécutives), + nouvelle métrique "phase decoding accuracy" (`argmax(γ_t)` vs `last_frame_phase` vérité terrain), bootstrap patient-level, Holm-Bonferroni contre `identity_dynamics` comme comparaison pré-enregistrée principale (le baseline le plus proche scientifiquement, §0).

**B. Qualité probabiliste** (nouveau — rien de comparable n'existe pour E1/GRU aujourd'hui, `metrics.py` n'a que `log_loss` binaire) :
- **Calibration** : reliability diagram (probabilité prédite vs fréquence observée, par bin), sur `γ_t(argmax)`.
- **Brier score multi-classe** : `(1/T)Σ_t Σ_i (γ_t(i) - 𝟙[S_t=i])²` — extension directe de ce que `metrics.py::compute_metrics` fait déjà en binaire (`log_loss`), à ajouter en **nouvelle fonction additive** dans `evaluation/metrics.py` (jamais en modifiant `compute_metrics` existant, pour ne pas casser la reproductibilité verrouillée d'E1/GRU qui en dépendent).
- **Log-vraisemblance** : `P(O_{1:T})` du forward, par trajectoire Test, agrégée — mesure la qualité du modèle génératif complet, pas seulement du décodage.
- **Cohérence des transitions** : fraction de trajectoires Viterbi (libre) violant la contrainte d'ordre (déjà en §4) — sert des deux dimensions à la fois (une violation fréquente indique un problème de performance, pas seulement de contrainte).
- **Qualité des durées** (si Semi-HMM, Phase 6) : comparaison de la distribution de durée *prédite* par état contre la distribution *réelle* observée sur Test (K-S test ou comparaison de quantiles), pas seulement une erreur moyenne.

**Règle explicite** : un HMM avec une meilleure AUROC mais une pire calibration (Brier/ECE) n'est **pas** strictement meilleur — les deux dimensions doivent être rapportées ensemble dans toute conclusion, jamais l'une sans l'autre (directive du prompt, cohérente avec la discipline "never fabricate results / always report full context" déjà établie dans `PROJECT_STATE.md` §8).

---

## 11. Étape 11 — Préservation et reproductibilité

Structure de fichiers proposée, cohérente avec la convention déjà en place (`Results/evaluation/e1_*`, `Results/evaluation/e1_gru_*`) :

```
Results/evaluation/
  e1_full_analysis/                    ← existant, verrouillé
  e1_frame_shuffle/                    ← existant, verrouillé
  linear_ssm_dynamics_report.json      ← existant, verrouillé
  e1_gru_step6a_full/                  ← existant, verrouillé
  e1_gru_step6b_test_evaluation/       ← existant, verrouillé
  e1_gru_step6c_frame_shuffle/         ← existant, verrouillé
  e1_hmm_step{N}_<étape>/              ← NOUVEAU, une sous-arborescence par étape numérotée (miroir exact du schéma GRU STEP1..6)
  e1_hmm_constrained_step{N}_<étape>/
  e1_semihmm_step{N}_<étape>/          ← seulement si Phase 6 (§5, §12) est GO
```

Chaque run enregistre (déjà le standard implicite d'E1/GRU, à rendre explicite en un petit `run_metadata.json` systématique, nouveau fichier, pas une modification d'existant) : config (hyperparamètres HMM : contraintes actives, `latent_dim` si réduction PCA réutilisée, distribution de durée si Semi-HMM), seed, split utilisé + son hash, checksum de l'embedding cache (`sha256_of_file`, déjà présent dans `embeddings/cache.py`, réutilisable tel quel), commit git, date, chemin des artefacts. Aucun résultat historique n'est jamais écrasé — chaque script `run_hmm_*` écrit sous son propre sous-dossier, jamais dans `e1_full_analysis/` ou `e1_gru_*`.

---

## 12. Étape 12 — Plan d'implémentation

| Phase | Objectif | Modifications autorisées | Fichiers | Tests | Résultats attendus | GO/NO-GO |
|---|---|---|---|---|---|---|
| **0** | Audit + extraction (fait dans ce document, §1) | Aucune | — | — | Table §1 | Fait — GO automatique vers Phase 1 |
| **1** | Données : joindre `embryo_dataset_time_elapsed` aux fenêtres déjà en cache (par `Video_name`+`Identifier`), sans toucher `preProcess.py`/`DataSet.py` | Nouveau fichier uniquement (ex. `Training/embeddings/join_elapsed_time.py`, composition sur le cache existant) | Nouveau script + nouveau champ optionnel dans `EmbeddingCache`/manifest (additif) | Test synthétique du join sur données factices | Table de durées réelles par fenêtre, prête pour §5 | GO si le join réussit sans erreur sur Train ; sinon Semi-HMM (Phase 6) reste bloquée mais HMM (Phase 2-4) peut continuer sans |
| **2** | HMM minimal (§3), état = phase (§2 option A/B) | Nouveau fichier `Training/evaluation/models/hmm.py` (`register_model("hmm")`), respecte l'interface `Model` | `evaluation/models/hmm.py`, `Tests/evaluation/models/test_hmm.py` | Unitaires synthétiques : `π`/`A` corrects sur séquence connue, `test_hmm_filtering_is_causal` (mirroir des tests GRU), forward/backward/Viterbi vérifiés à la main sur un petit exemple | Log-vraisemblance Test, decoding accuracy vs `identity_dynamics` | GO vers Phase 3 si le code passe tous les tests et tourne sur un sous-ensemble réel (mirroir du STEP 3 GRU, smoke test CPU) |
| **3** | HMM contraint (§4) | Extension de `hmm.py` (paramètre `constrained: bool`, pas un nouveau fichier) | idem + tests de violation de contrainte | Test : `A` contraint a bien des zéros structurels ; aucune transition interdite dans le décodage Viterbi contraint | Comparaison quantitative libre vs contraint (§4) | GO vers Phase 4 si le contraint ne dégrade pas significativement la vraisemblance |
| **4** | Reporting probabiliste (§7) + comparaison complète vs E1/GRU (§6) | Nouveau fichier `Training/evaluation/run_hmm_test_evaluation.py` (mirroir de `run_gru_test_evaluation.py`), + fonctions additives dans `metrics.py` (Brier multi-classe, entropie) | `run_hmm_evaluation.py`, `run_hmm_test_evaluation.py`, extension additive de `metrics.py` | Régression : 79 tests existants toujours verts après extension de `metrics.py` | `REPORT.md` HMM (libre + contraint) vs les 6 lignes E1/GRU déjà mesurées | **Décision utilisateur explicite requise avant de lancer sur GPU/Test**, comme pour chaque STEP GRU |
| **5** | Frame-shuffle spécifique HMM (§6, note sur l'interprétation inversée) | Nouveau fichier | `run_hmm_frame_shuffle_sanity_check.py` | — | Contrôle de validité adapté (§6) | GO/NO-GO scientifique sur le HMM contraint |
| **6** | Analyse des durées (§5, critère GO/NO-GO explicite) | Aucun code modèle, analyse statistique seule | Notebook/script d'analyse, pas encore de modèle | — | Verdict géométrique-suffisant vs Semi-HMM-justifié | **NO-GO par défaut** — ne passer à la Phase 7 que si le test décrit en §5 montre un écart significatif |
| **7** | Semi-HMM (§5), **seulement si Phase 6 = GO** | Nouveau fichier `Training/evaluation/models/semi_hmm.py` | idem structure hmm.py | Unitaires + comparaison durée prédite vs réelle | Table durée par phase avec IC | GO/NO-GO sur la qualité de calibration des durées (§10B) |
| **8** | Validation probabiliste complète (§10) | Extension additive de `metrics.py`/nouveau `evaluation/calibration.py` | — | Tests de calibration sur données synthétiques à calibration connue | Verdict combiné performance + calibration | Rapport final de l'étude |
| **9** | Schéma JSON RAG (§8) figé | Nouveau fichier `evaluation/reporting_schema.py` (dataclass/validation, pas de génération de texte) | — | Validation de schéma | Schéma stable, versionné | Prêt pour intégration RAG future — **hors scope de ce plan** |
| **10, 11** | RAG, application web | Hors scope de ce plan (cohérent avec `RESEARCH_BLUEPRINT.md` Part V "NOT NOW") | — | — | — | Non planifiées ici |

Chaque phase GPU/Test (4, 5, 7) suit la même discipline d'autorisation explicite déjà en place pour E1/GRU (`PROJECT_STATE.md` §10, point 7) : pas de lancement automatique, vérification `nvidia-smi`/`ps aux`/`screen -ls` avant tout usage, jamais sur un GPU déjà occupé.

---

## 13. Étape 13 — Conclusion de l'étude

1. **Le HMM est-il adapté ?** Oui, sous la forme (a) §0 (décodage contraint, état = phase observée) — c'est littéralement un baseline déjà identifié comme manquant dans `RESEARCH_BLUEPRINT.md` Part VIII, indépendant du gate RQ1-négatif. La forme (b) (régime latent = E2/H2) est adaptée en théorie mais **gatée** par le résultat négatif de RQ1 — nécessite une décision explicite de réouverture, pas un GO automatique.
2. **Quelle définition des états est la plus pertinente ?** Phase biologique (`chronological_phases`, déjà dans `first_frame_phase`/`last_frame_phase`) — gratuite, interprétable, même supervision que E1/GRU, zéro nouveau travail d'étiquetage.
3. **Quelles observations utiliser ?** Les embeddings `resnet18_balanced` déjà en cache (512D), via une émission hybride discriminative (régression logistique multinomiale convertie par Bayes), pas un mélange de gaussiennes en haute dimension.
4. **Quelles contraintes sont justifiables ?** Ordre chronologique et absence de retour en arrière (`A_{ij}=0` hors `i`, `i+1`) — directement issues de `HANDOFF.md` §4.6 et de la logique déjà codée dans `DataSet.py`, pas inventées. À présenter comme un choix de modélisation motivé par la définition même des phases, pas comme un résultat empirique (§2, note de circularité).
5. **HMM ou Semi-HMM d'abord ?** HMM (contraint) d'abord, dans tous les cas. Le passage au Semi-HMM est conditionné à un test statistique explicite (§5) sur des durées réelles qui n'existent pas encore dans le pipeline (le join Phase 1 est un préalable strict, pas optionnel).
6. **Quelles informations pour un reporting fiable ?** Distinguer strictement filtrage (causal) vs lissage/Viterbi (rétrospectif), et probabilité vs score/vraisemblance non normalisée (§7) — la majorité du risque méthodologique de cette étude est ici, pas dans le choix du modèle lui-même.
7. **Comment comparer proprement à E1/Linear SSM/GRU ?** Réutiliser tel quel `evaluation.trajectory`/`evaluation.model`/`evaluation.metrics` (déjà 100% compatibles, aucune modification requise pour l'intégration de base), même bootstrap/Holm-Bonferroni, en acceptant explicitement que certaines colonnes n'ont pas de sens pour tous les modèles (§6) plutôt que de forcer une comparaison artificielle.
8. **Quel format pour le RAG ?** Le schéma JSON §8, où chaque champ numérique porte son `type` sémantique — condition nécessaire pour que le RAG ne recalcule ni ne confonde jamais une probabilité et un score.
9. **Prototype minimal ?** Phase 2 seule (§12) : HMM non contraint, état=phase, émission hybride logistique, évalué en decoding-accuracy + log-vraisemblance sur un sous-ensemble réel (mirroir du STEP 3 GRU) — sans toucher GPU/Test.
10. **Quels résultats concluraient que cette approche est réellement meilleure (vs juste différente) ?** Un gain significatif (bootstrap + Holm-Bonferroni) en decoding-accuracy/recall-transition par rapport à `identity_dynamics`, qui **survit** à un contrôle de shuffle adapté (§6 — pas la même lecture que pour GRU), **et** une calibration (Brier/ECE) au moins équivalente — les trois conditions ensemble, pas une seule métrique favorable. Un gain uniquement en interprétabilité (sans gain de performance ni de calibration) est une conclusion valide mais différente : "différent", pas "meilleur" — à rapporter honnêtement comme tel, dans la continuité de la discipline déjà appliquée aux résultats négatifs d'E1 et de GRU.
