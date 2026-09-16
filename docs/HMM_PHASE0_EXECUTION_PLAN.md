# HMM — Plan d'exécution concret, Phase 0

Écrit 2026-08-19. Complément exécutable de `docs/HMM_RESEARCH_PLAN.md` (théorie, math, schéma JSON déjà spécifiés là-bas — non répétés ici sauf pour préciser des points concrets). Aucune commande d'écriture, d'entraînement, ou de modification n'a été exécutée pour produire ce document — audit en lecture de code source local uniquement (`Training/embeddings/`, `Training/evaluation/`, `Training/DataSet.py`, `Training/preProcess.py`). Aucune donnée réelle (`Data/`, `Embeddings/`, `Results/`) n'existe dans ce checkout local — voir "Informations manquantes" partout où c'est pertinent, elles nécessitent un accès au serveur GPU (`gpu1.labisen.isen-ouest.fr`, lecture seule) pour être levées.

---

## 1. Étape 1 — Audit complet des données (précis, par fichier)

### Localisation exacte de chaque objet demandé

| Objet | Fichier / emplacement exact |
|---|---|
| Embeddings | `Embeddings/{model_name}/{split}/embeddings.pt` — tenseur `(N, 512)` float32 (`Training/embeddings/cache.py:6-8`) |
| Metadata | `Embeddings/{model_name}/{split}/metadata.csv` — colonnes ci-dessous |
| Manifest (provenance) | `Embeddings/{model_name}/manifest.json` (partagé) + `Embeddings/{model_name}/{split}/manifest.json` (par split) — `model_name, checkpoint_path, checkpoint_sha256, window_size, stride, focal_type, image_size, mode, extractor_version, created_at` (`Training/embeddings/cache.py:46-58`) |
| Construction des fenêtres | `Training/DataSet.py::Embryo_Transition_Dataset._create_sequences` (lignes 188-221) |
| Phases (source) | `Training/DataSet.py:151-167`, `chronological_phases` (15 valeurs, `tHB` exclue ligne 123) |
| Labels par fenêtre | Produits par `Embryo_Transition_Dataset.__getitem__` (lignes 266-317), capturés dans `Training/embeddings/build_cache.py:415-429` |
| Timestamps réels | `Data/embryo_dataset_time_elapsed/{Patient_N}_timeElapsed.csv` — **jamais lu par aucun code du repo** (voir §2) |
| Résultats E1 | `Results/evaluation/e1_full_analysis/`, `e1_frame_shuffle/`, `linear_ssm_dynamics_report.json` (remote uniquement — voir `docs/PROJECT_STATE.md` §6) |
| Résultats GRU | `Results/evaluation/e1_gru_step6{a,b,c}_*/` (remote uniquement) |
| Tests existants | `Tests/evaluation/models/test_{base_rate,identity_dynamics,persistence,linear_ssm,gru}.py`, `Tests/evaluation/test_metrics.py` (nom exact à confirmer par `ls`), `Tests/conftest.py` |

### Colonnes exactes de `metadata.csv` (confirmé par lecture de `build_cache.py:420-429` + `embeddings/dataset.py:82-88`)

```
embedding_index    int, ajouté au merge final (build_cache.py:501), unique, aligné 1:1 avec la ligne i de embeddings.pt
video_name         str, ex. "Patient_5"
window_start       int  — voir définition exacte ci-dessous, PIÈGE
consistency_flag   int {0,1}
first_frame_phase  int  — index de phase, PIÈGE (voir ci-dessous)
last_frame_phase   int  — index de phase, PIÈGE (voir ci-dessous)
```

Il n'y a **aucune colonne** de temps réel, de nom de fichier image, ou de numéro de frame brut dans `metadata.csv` tel qu'écrit aujourd'hui.

### `window_start` — définition exacte, pas celle qu'on croit intuitivement

Ce n'est **pas** "le numéro de la frame de départ dans la vidéo". C'est l'**index global** dans `dataset.video_sequences` (la liste plate de toutes les fenêtres de **tout le split**, toutes vidéos confondues), tel que produit par `_video_order_and_indices` (`build_cache.py:240-260`) puis directement assigné (`build_cache.py:424`, `"window_start": int(orig_indices[i])`). Preuve indépendante : `embeddings/validate.py:380-381` fait `item = dataset[window_start]` pour son spot-check de déterminisme — cela n'a de sens que si `window_start` est bien un index direct dans `dataset.video_sequences`, pas un offset de frame local à la vidéo.

Conséquence pratique : pour Video B qui suit Video A dans l'ordre de construction (492 vidéos pour Train), les `window_start` de Video B ne commencent pas à 0 mais reprennent après le dernier index de Video A. C'est **contigu par vidéo** (confirmé empiriquement, commentaire `build_cache.py:246-251`) et **monotone croissant dans l'ordre temporel réel** au sein d'une même vidéo (les fenêtres sont créées dans `_create_sequences` avec `start` croissant, stride constant) — donc le tri par `window_start` au sein d'un groupe `video_name` (déjà fait par `evaluation/trajectory.py::group_into_trajectories`, ligne 71) reste correct. Mais la valeur absolue ne représente aucune position temporelle interprétable seule — **c'est exactement le problème pour le join temporel (§2)**.

Récupération de l'offset local (sans toucher au code existant, pure lecture) : pour chaque vidéo, `local_start = window_start - min(window_start du groupe)`. Le vrai numéro de frame de la première image de la fenêtre est alors `local_start * stride` **dans la liste triée des frames de cette vidéo telle que `_create_sequences` la construit** — pas nécessairement un numéro de frame brut contigu (des frames peuvent manquer suite à `remove_images_not_in_csv`/`remove_problematic_images` dans `preProcess.py`).

### `first_frame_phase` / `last_frame_phase` — définition exacte, et un risque non détecté avant ce document

Ce sont les indices (`int`) dans `Embryo_Transition_Dataset.phase_to_index`, dict construit ligne 172-174 :
```python
present_phases = [p for p in chronological_phases if p in self.data_frame[phase_col].unique()]
phase_to_index = {label: index for index, label in enumerate(present_phases)}
```

**`present_phases` dépend des phases effectivement présentes dans CE split** (`self.data_frame` est déjà filtré par `UsedFor == split` avant ce calcul, ligne 122). `build_cache.py` instancie un `Embryo_Transition_Dataset` séparé par split (`main()`, ligne 300-308, un appel par valeur de `--split`). **Rien ne garantit que `phase_to_index` est identique entre Train/Val/Test** si une phase rare (ex. `tPB2`, `tEB`) est absente d'un split — le repo ne sauvegarde `phase_to_index` nulle part dans `metadata.csv`/`manifest.json` (seul `print()` à la construction, jamais capturé dans un fichier selon ce qui est visible localement).

**Conséquence si c'est effectivement le cas** : l'entier `3` dans `first_frame_phase` du split Train pourrait désigner une phase différente de l'entier `3` dans Test — silencieusement faux pour tout usage du HMM qui traite ces entiers comme un espace d'états partagé entre splits (exactement notre cas). **Ceci n'a jamais posé de problème pour E1/GRU** car ces modèles n'utilisent que `consistency_flag` (comparaison intra-fenêtre, insensible à la valeur absolue de l'index) — c'est un risque **nouveau, introduit par cette branche HMM**, jamais testé auparavant.

**Information manquante, à vérifier avant toute chose (Phase 0 concrète, §12)** : reconstruire (lecture seule, pas de GPU, `Embryo_Transition_Dataset.__init__` ne charge aucune image — seulement `pandas.read_csv` + groupby, quelques secondes) les trois datasets Train/Val/Test avec les arguments exacts enregistrés dans `Embeddings/resnet18/manifest.json`, et comparer leurs `.phase_to_index`. C'est la vérification la plus urgente de tout ce plan — voir §14.J.

### Nombres exacts — ce qui est déjà mesuré vs ce qui manque

**Déjà mesuré** (source : `docs/PROJECT_STATE.md` §3a, sessions précédentes, confirmé remote) :
- Dimension embedding : 512 (resnet18, pénultième couche)
- Vidéos : Train 492 / Val 106 / Test 106
- Fenêtres : Train 196934 / Val 43339 / Test 42519
- Checkpoint : `resnet18_balanced`, SHA-256 `be5460d0...`
- T (fenêtres par trajectoire) : Train min 83/médiane 431/max 618 ; Val min 100/médiane 428/max 627 ; Test min 119/médiane 433/max 553

**Manquant, non calculé nulle part dans le repo actuel** (à produire, lecture seule, §12 Phase 0/2) :
- Nombre de fenêtres **par phase**, par split
- Nombre de trajectoires contenant chaque phase
- Première/dernière apparition de chaque phase (en index de fenêtre — pas encore en temps réel, voir §2)
- Matrice de transition empirique complète (l'équivalent de `_print_transition_matrix` de `DataSet.py`, mais celui-ci n'est imprimé qu'en stdout à la construction — jamais sauvegardé en fichier, et raisonne sur `(first_phase, last_phase)` par fenêtre, pas sur la séquence de fenêtres elle-même)
- Retours en arrière observés dans les données réelles (à distinguer des retours en arrière qui seraient déjà structurellement impossibles à cause du filtre d'adjacence de `_create_sequences`, voir §6)
- Proportion de données manquantes (frames retirées par `remove_images_not_in_csv`/`remove_problematic_images` dans `preProcess.py`, impact sur la continuité des fenêtres) — non quantifié

---

## 2. Étape 2 — Join temporel : mécanisme exact, pas encore réalisable tel quel

### Ce qu'on sait avec certitude sur `embryo_dataset_time_elapsed`

Le seul code qui y touche est `Training/preProcess.py::rename_shared_csvs` (lignes 113-120) : chaque fichier est renommé `{OldPatientName}_timeElapsed.csv` → `{NewPatientName}_timeElapsed.csv`, **jamais ouvert, jamais parsé**. Donc :
- **Information manquante — colonnes exactes de `{Patient}_timeElapsed.csv`** : inconnues du code. Ni le nom des colonnes, ni l'unité (heures ? minutes ? secondes depuis fécondation ?), ni la clé (numéro de frame ? nom de fichier ? horodatage absolu ?) ne sont déterminables par lecture de ce repo. **Ne pas supposer un format — ouvrir un exemplaire réel du fichier sur le serveur (`gpu1.labisen.isen-ouest.fr`) ou la sauvegarde NAS avant de concevoir le join en détail.**
- **Information manquante — granularité** : un timestamp par frame (le plus probable, cohérent avec le time-lapse) ou par phase (moins probable, redondant avec `_phases.csv`) — à vérifier par la même lecture directe.

### Mécanisme de jointure recommandé (une fois le format confirmé)

Trois sauts, tous en lecture seule, aucun ne nécessite de modifier `metadata.csv` ni `DataSet.py` :

1. **`metadata.csv` row → fenêtre réelle** : reconstruire `Embryo_Transition_Dataset` avec les arguments exacts du manifest (`window_size`, `stride`, `focal_type`, split) et indexer `dataset.video_sequences[window_start]` — **c'est exactement le mécanisme déjà utilisé par `embeddings/validate.py:380-397`** pour son spot-check de déterminisme, réutilisable tel quel (composition, zéro nouveau code de reconstruction de fenêtre à écrire). Cela donne la liste des `window_size` dicts `{identifier, path, phase}` réels de la fenêtre — première et dernière frame incluses.
2. **Identifiant de frame → temps réel** : joindre `identifier`/le numéro de frame extrait (même regex que `DataSet.py::_create_sequences` ligne 202-204, `re.findall(r"\d+", ...)`) contre `{video_name}_timeElapsed.csv` — mécanisme exact dépendant du format confirmé à l'étape précédente.
3. **Choix du point d'alignement temporel** : recommandation (cohérente avec `docs/HMM_RESEARCH_PLAN.md` §3, alignement `last_frame_phase`) — utiliser le temps de la **dernière** frame de la fenêtre comme timestamp de la fenêtre, pour rester cohérent avec un état "tel qu'il est à la fin de la fenêtre la plus récente", utilisable en filtrage causal.

### Analyse demandée (§2 du prompt) — ce qui est vérifiable dès maintenant vs ce qui attend le format réel

| Point demandé | Vérifiable maintenant ? |
|---|---|
| Clé de jointure | Partiellement — le côté `metadata.csv`/`Splits/{Focal}.csv` est clair (`Identifier`, regex numérique) ; le côté `timeElapsed.csv` est inconnu |
| Granularité | Inconnue — dépend du fichier réel |
| Unités | Inconnue — dépend du fichier réel |
| Timestamps manquants | Non vérifiable sans le fichier réel |
| Doublons | Non vérifiable sans le fichier réel |
| Fenêtres sans correspondance | Non vérifiable, mais **structurellement possible** : `remove_images_not_in_csv`/`remove_problematic_images` (`preProcess.py`) suppriment des images du pipeline principal sans qu'on sache si `embryo_dataset_time_elapsed/` a été nettoyé en miroir — à vérifier explicitement, ne pas supposer une correspondance parfaite |
| Correspondances multiples | Non vérifiable sans le fichier réel |
| Monotonicité du temps | Non vérifiable sans le fichier réel — mais **attendue** vu qu'il s'agit d'un enregistrement time-lapse séquentiel ; à confirmer, pas à supposer |
| Différences entre vidéos | Non vérifiable — mais plausible (durée totale d'acquisition différente par embryon, déjà visible indirectement dans la variance du nombre de fenêtres par trajectoire, T min=83/max=618 sur Train) |

### Fichier de metadata enrichie — colonnes proposées (description uniquement, rien n'est créé)

```
Embeddings/{model_name}/{split}/metadata_with_time.csv
```
en plus des colonnes existantes de `metadata.csv` (jamais modifiées) :

```
first_frame_identifier   str  — ex. "Patient_5_Image_142.jpeg" (recouvré via le saut 1 ci-dessus)
last_frame_identifier    str
first_frame_elapsed_time float, unité à confirmer (§2) — NaN si non trouvé, JAMAIS interpolé silencieusement
last_frame_elapsed_time  float, idem — c'est cette colonne qui sert de timestamp de fenêtre par défaut (voir choix d'alignement)
elapsed_time_source_row  str/int — traçabilité vers la ligne source du fichier timeElapsed, pour audit
join_status               str enum {"ok", "missing_in_time_csv", "duplicate_in_time_csv", "video_missing_time_csv"} — explicite, jamais un NaN silencieux sans raison
```

`metadata.csv` original n'est jamais réécrit — `metadata_with_time.csv` est un fichier strictement additif, à côté, produit par un nouveau script (`Training/embeddings/join_elapsed_time.py`, jamais écrit à ce stade).

---

## 3. Étape 3 — Analyse des phases : calculs à effectuer (aucun lancé)

Tous ces calculs sont read-only, sur `metadata.csv` + reconstruction `Embryo_Transition_Dataset` (§1), pas sur les images — donc rapides (secondes à quelques minutes), pas besoin de GPU.

**Par phase, par split** :
- `n_windows_first = metadata.groupby("first_frame_phase").size()`, idem pour `last_frame_phase`
- `n_trajectories = metadata.groupby("first_frame_phase")["video_name"].nunique()`
- Première/dernière apparition : `groupby("video_name")` puis min/max de `window_start` (normalisé en local, §1) par phase — nécessite l'alignement local par vidéo décrit en §1
- Durée observée : nécessite §2 (temps réel) — sinon seulement une durée en "nombre de fenêtres", explicitement à étiqueter comme non clinique tant que §2 n'est pas résolu

**Transitions observées** :
1. **Matrice de transition empirique au niveau fenêtre** : reproductible directement, c'est `DataSet.py::_print_transition_matrix` (lignes 250-261) mais **appliquée sur `metadata.csv` en cache plutôt qu'en relançant `Embryo_Transition_Dataset`** (équivalent, car les mêmes `first_frame_phase`/`last_frame_phase` sont déjà dedans) — à sauvegarder en fichier cette fois (jamais fait jusqu'ici).
2. **Matrice de transition au niveau trajectoire (phase → phase suivante distincte)** : différente de (1) — nécessite de dérouler chaque trajectoire triée (`evaluation.trajectory.group_into_trajectories`, déjà existant, réutilisable tel quel), extraire la séquence de `last_frame_phase` compressée en "run-length" (phases consécutives identiques fusionnées), puis compter les bigrammes de cette séquence compressée — c'est **la vraie matrice `A` du HMM au niveau état/temps de fenêtre**, distincte de (1) qui ne regarde que le début/fin d'une seule fenêtre.
3. **Retours en arrière observés** : sur la séquence compressée de (2), compter les paires `(phase_i, phase_j)` avec `index(phase_j) < index(phase_i)` dans `chronological_phases` — **distinction importante** : `_create_sequences` interdit déjà les retours en arrière *au niveau d'une fenêtre individuelle* (adjacence stricte, `DataSet.py:218-220`), mais rien n'empêche une trajectoire de revenir en arrière *entre deux fenêtres non consécutives dans le temps* si l'étiquetage humain original contient une inversion (annotateur, erreur de phase). Ce calcul est donc réellement informatif, pas tautologique — contrairement à la contrainte au niveau fenêtre.
4. **Observations incohérentes** : fenêtres où `first_frame_phase`/`last_frame_phase` ne sont pas adjacentes dans `chronological_phases` — **ne devraient structurellement pas exister** vu le filtre de `_create_sequences` ; leur présence éventuelle indiquerait soit un bug soit une différence entre le `chronological_phases` utilisé à la construction du cache et celui utilisé pour l'analyse (encore une raison de vérifier §1's `phase_to_index`).
5. **Proportion de données manquantes** : fraction de fenêtres candidates (`len(sorted_frames) - window_size + 1` par vidéo) effectivement présentes dans le cache, vs celles perdues à cause d'échecs de chargement d'image (`build_cache.py:433-437`, déjà loggé en `WARNING` dans les logs d'extraction si conservés) — à retrouver dans les logs remote existants plutôt qu'à recalculer.

**Livrables de cette étape (fichiers à produire, pas encore produits)** :
1. Matrice de transition empirique (niveau fenêtre ET niveau trajectoire compressée — deux matrices distinctes, §3.2)
2. Distribution des durées par phase (bloquée par §2 pour l'unité réelle ; calculable dès maintenant en "nombre de fenêtres")
3. Graphe des transitions observées (visualisation de la matrice (2) ci-dessus, nœuds = phases présentes, arêtes = transitions avec compte > 0)
4. Histogramme des durées par phase (idem, bloqué par §2 pour l'unité réelle)

---

## 4. Étape 4 — HMM minimal : décision d'émission

Renvoi à `docs/HMM_RESEARCH_PLAN.md` §3 pour la formulation complète (`π`, `A`, algorithmes). Ici, l'analyse comparative des émissions demandée explicitement :

| Option | Hypothèse | Paramètres à estimer | Risque principal | Adapté à 512D avec ~500 trajectoires ? |
|---|---|---|---|---|
| **Gaussian HMM plein 512D** (`Σ` pleine par état) | Émission gaussienne multivariée par état | `15 × (512 + 512×513/2)` ≈ 15 × 131 585 ≈ **2M paramètres** pour `Σ` seule | Surapprentissage massif — largement plus de paramètres que de fenêtres d'entraînement disponibles pour les phases rares | **Non** |
| **Gaussian HMM diagonale (`Σ` diagonale)** | Composantes de l'embedding indépendantes conditionnellement à l'état | `15 × 512 × 2` (moyenne + variance) ≈ 15 360 | Toujours beaucoup, et l'hypothèse d'indépendance des 512 dimensions d'un embedding CNN est peu plausible | Limite, à tester seulement si (b) échoue |
| **PCA + Gaussian** | Réduction à `latent_dim` (ex. 16, comme `linear_ssm`) avant émission gaussienne | `15 × (latent_dim + latent_dim²/2)`, ex. `latent_dim=16` → ≈ 2 160 | Le choix de `latent_dim`/la PCA elle-même perd de l'information — mais déjà la stratégie validée pour `linear_ssm` (`Training/evaluation/models/linear_ssm.py`, PCA `svd_solver="full"`), cohérence méthodologique avec l'existant | Oui, raisonnable |
| **Régression logistique + transformation probabiliste (hybride discriminatif)** | `P(S=i\|O)` appris directement (discriminatif), converti en pseudo-vraisemblance par Bayes `P(O\|S=i) ∝ P(S=i\|O)/P(S=i)` | Identique à une régression logistique multinomiale 512D → 15 classes, `512×15` ≈ 7 680 coefficients, déjà le même ordre de grandeur et la même brique que `identity_dynamics`/`persistence`/`linear_ssm` (`sklearn.linear_model.LogisticRegression`) | Les probabilités hybrides ne sont pas de vraies vraisemblances calibrées sans vérification (nécessite un contrôle de calibration, §10 du plan conceptuel) — mais c'est un défaut connu et documenté de la technique (Bourlard & Morgan), pas une surprise | **Oui — recommandé** |
| **GMM par état (mélange de gaussiennes)** | Densité multimodale par état | `K_mixtures × (15 × (D + D²/2))` — pire que Gaussian plein si `D=512` | Combine les deux problèmes précédents (trop de paramètres + choix arbitraire de `K_mixtures`) | Non, sauf en combinaison avec PCA (alors équivalent à "PCA + GMM", une variante de l'option 3) |

**Recommandation inchangée par rapport à `docs/HMM_RESEARCH_PLAN.md` §3** : hybride discriminatif (option 4) en premier prototype (le plus proche de l'infrastructure `sklearn` déjà utilisée trois fois dans ce repo, le moins de nouveaux paramètres, le plus rapide à valider) ; PCA+Gaussian (option 3) comme deuxième candidat si l'hypothèse discriminative-vers-générative pose un problème de calibration en pratique (à mesurer, pas à supposer).

---

## 5. Étape 5 — Première sortie attendue : distinctions mathématiques (rappel opérationnel)

Renvoi complet à `docs/HMM_RESEARCH_PLAN.md` §7 pour la taxonomie détaillée. Rappel condensé, appliqué à l'exemple du prompt :

| Terme du prompt | Nom mathématique exact | Utilise le futur ? |
|---|---|---|
| "État le plus probable" | `argmax_i γ_t(i)` avec `γ_t` = **posterior de filtrage** (`α_t` normalisé), PAS lissage | Non (si filtrage) |
| "Posterior des états" | `γ_t(i) = α_t(i) / Σ_j α_t(j)` (filtrage) | Non |
| "Prochaine transition probable" | `argmax_j [Σ_i γ_t(i) A_{ij}]` — **distribution prédictive**, pas la ligne `A` de l'état arg-max seul | Non |
| "Probabilités des transitions possibles" | `Σ_i γ_t(i) A_{ij}` pour tout `j` | Non |
| (non demandé ici mais à ne jamais confondre avec ce qui précède) "Trajectory probability" | Score Viterbi normalisé ou probabilité de segmentation sous contrainte left-to-right | **Oui — rétrospectif, jamais pour "l'état actuel"** |

Le "Temps : 42.3h" de l'exemple du prompt dépend de §2 (join temporel) — sans lui, seul un index de fenêtre est reportable, pas une heure réelle. À afficher comme tel dans tout prototype tant que §2 n'est pas résolu (ex. `"window_index": 214` plutôt que `"time_h": 42.3`).

---

## 6. Étape 6 — HMM contraint : classification des contraintes par niveau de preuve

| Contrainte | Démontrée par les données actuelles ? | Justifiée par la documentation du repo ? | Justifiée par la littérature ? | À valider avec la professeure |
|---|---|---|---|---|
| Maintien dans la phase (`A_ii` libre) | Oui (déjà la valeur par défaut, aucune contrainte à ajouter) | — | — | Non nécessaire |
| Transition vers le successeur immédiat | **Partiellement tautologique** — déjà imposé au niveau fenêtre par `DataSet.py::_create_sequences` (adjacence stricte), donc "démontré" en partie par construction du dataset, pas uniquement par la biologie observée. Le test réel (retours en arrière au niveau trajectoire compressée, §3.3) reste à faire | Oui — `chronological_phases` est déjà l'ordre documenté du repo | Oui — c'est la définition standard de la nomenclature morphocinétique (Meseguer et al., déjà la base du jeu de données Gomez et al. cité dans `HANDOFF.md` §1) | Recommandé — confirmer que sauter directement une phase (ex. t2→t4 sans t3 observé) est bien biologiquement impossible et pas juste rare/mal annoté |
| Transitions interdites (saut de plus d'une phase) | Non testé sur données réelles à ce stade (§3.4, "observations incohérentes") | Oui, cohérent avec le filtre déjà en place | Oui | Recommandé — même remarque |
| Retours en arrière interdits | **Non démontré** — c'est précisément le calcul §3.3 qui reste à faire ; ne pas l'imposer avant de savoir si des cas réels existent (erreurs d'annotation légitimes à conserver comme telles, pas à masquer par une contrainte trop rigide — cohérent avec `HANDOFF.md` §4.6 : "real biology can violate [soft constraints] in informative ways") | Partiellement (`HANDOFF.md` §4.6 classe l'irréversibilité comme contrainte **dure**, mais au niveau du régime `r(t)`, pas nécessairement au niveau de l'étiquetage humain de phase, qui peut contenir du bruit d'annotation) | Oui en théorie biologique, mais l'annotation humaine n'est pas nécessairement sans erreur (`HANDOFF.md` §8 : "existing phase labels... not verified — inter-observer variability... is real and documented") | **Oui, obligatoire** — décider si on modélise l'irréversibilité biologique (contrainte dure sur le process réel) ou si on doit tolérer un bruit d'annotation résiduel (contrainte souple/pénalité, pas un `A_ij=0` strict) |
| États initiaux (restreints aux phases observées en première fenêtre) | Oui, dérivable empiriquement de `π` | — | — | Non nécessaire |
| États terminaux (structure left-to-right stricte) | À décider — dépend de si on veut la propriété de tractabilité en §7 du plan conceptuel (segmentation) | Cohérent avec `HANDOFF.md` §4.6 | Oui, structure standard "Bakis"/left-to-right en reconnaissance de séquence | Optionnel, discussion de conception plutôt que de biologie |

**Point de méthode explicite (répété du plan conceptuel, important)** : ne pas présenter la contrainte d'ordre comme un résultat "découvert" par le HMM — elle est en grande partie déjà **imposée par construction** du dataset (`_create_sequences`). La vraie question empirique ouverte est le taux de retours en arrière **au niveau trajectoire compressée** (§3.3), qui n'est actuellement pas connu et n'est pas trivialement nul.

---

## 7. Étape 7 — Comparaison aux anciens résultats : plan, sans toucher aux artefacts

Reprend le tableau de `docs/HMM_RESEARCH_PLAN.md` §6, avec la clarification demandée sur les métriques réellement comparables :

**Directement comparables (même tâche, même infrastructure `evaluation.metrics`, aucune modification requise)** : accuracy/AUROC/recall sur `consistency_flag` (dérivable du décodage HMM : changement d'état décodé entre `t-1` et `t`), via `evaluation.metrics.compute_metrics` + `bootstrap_trajectory_metrics` + `paired_bootstrap_comparison` + `holm_bonferroni`, tous réutilisables sans modification (confirmé par lecture de `Training/evaluation/model.py`/`metrics.py`/`trajectory.py` — l'interface `Model.predict() -> List[TrajectoryPrediction]` ne présuppose rien sur le mécanisme interne du modèle).

**Non directement comparables, à traiter séparément, pas à forcer dans le même tableau** :
- **Log-likelihood** : seul `linear_ssm`/GRU (formulation `residual`) ont une notion de vraisemblance/erreur de reconstruction ; `identity_dynamics`/`persistence`/`base_rate` n'en ont pas de comparable — le HMM en a une (`P(O_{1:T})`) mais sur une échelle et une définition différentes (vraisemblance de séquence d'état, pas erreur de régression d'embedding).
- **Calibration/Brier score** : **n'existe pour aucun modèle actuel** (`metrics.py` n'a que `log_loss` binaire) — nouvelle capacité à ajouter en fonctions additives dans `metrics.py`, jamais en modifiant les fonctions existantes (`compute_metrics`, dont dépendent les résultats E1/GRU verrouillés).
- **Qualité des trajectoires / respect des contraintes** : propre au HMM (et à GRU dans une moindre mesure, via ses tests de causalité) — pas de valeur comparable pour `base_rate`/`identity_dynamics`/`persistence`/`linear_ssm`, qui ne produisent pas de séquence d'état.
- **"Transition accuracy" vs "transition recall"** : déjà rapportés par E1/GRU sous les noms `recall`/`precision` sur la classe `consistency_flag=1` (`compute_metrics`) — même métrique, juste vérifier qu'on utilise le nom déjà standardisé du repo plutôt que d'en inventer un nouveau.

**Aucun artefact existant n'est touché par cette comparaison** — un nouveau script (`run_hmm_test_evaluation.py`, jamais encore écrit) chargerait les 4+2 modèles existants pour ré-ajuster/recharger (mirroir exact de ce que fait déjà `run_gru_test_evaluation.py`, §11) sans jamais réécrire dans `Results/evaluation/e1_*`/`e1_gru_*`.

---

## 8. Étape 8 — Analyse des durées : calculs précis, critère GO/NO-GO

Bloquée tant que §2 (join temporel) n'a pas confirmé le format réel de `embryo_dataset_time_elapsed`. Plan de calcul, une fois débloqué :

- `D` par occurrence de phase = temps réel de sortie de phase − temps réel d'entrée de phase, dérivé de la séquence compressée (§3.2) + `metadata_with_time.csv` (§2).
- **Nombre d'observations disponibles par phase** : nombre de trajectoires Train où cette phase apparaît avec une durée entièrement mesurée (pas censurée).
- **Durées censurées** : phase encore en cours à la fin de l'enregistrement (dernière fenêtre de la trajectoire) ou en tout début (première phase, dont l'entrée réelle précède potentiellement le début de l'enregistrement) — à traiter comme censure à droite/gauche, jamais comme une durée complète par défaut (biais classique sinon).
- **Trajectoires incomplètes** : vidéos où le nombre de fenêtres est anormalement bas (T min=83 sur Train, très inférieur à la médiane 431) — possible arrêt d'enregistrement précoce, à croiser avec les phases atteintes.
- **Variance/forme** : par phase, ajustement Gamma/log-normale vs géométrique implicite (test AIC/BIC ou QQ-plot), exactement le critère déjà posé dans `docs/HMM_RESEARCH_PLAN.md` §5.
- **Phases suffisamment représentées** : seuil à définir explicitement avant de regarder les résultats (ex. ≥30 occurrences non censurées) — pré-enregistré, pas choisi post-hoc, pour éviter le même genre de risque de circularité que celui déjà discuté en §2/§6 de `HMM_RESEARCH_PLAN.md`.

**Critère GO/NO-GO, répété explicitement** : GO vers Semi-HMM seulement si (a) §2 est résolu (unités réelles disponibles) ET (b) au moins une phase à représentation suffisante montre un écart significatif (AIC/BIC ou test de forme) entre géométrique et une distribution à mode non nul. NO-GO par défaut sinon — rester sur HMM contraint.

---

## 9-10-11. Étapes 9-10 — Reporting et interface RAG

Déjà spécifiés en détail dans `docs/HMM_RESEARCH_PLAN.md` §7 (format texte, taxonomie complète des quantités) et §8 (schéma JSON avec `type` sémantique par champ). Rien à ajouter de plus concret tant que §2 (temps réel) et le prototype §4/§12-Phase 3 n'existent pas — le format y est déjà défini au niveau nécessaire pour l'implémentation ; le répéter ici serait dupliquer une source de vérité déjà écrite. Seule différence à noter : tant que §2 n'est pas résolu, tout champ `duration`/`estimated_duration` du schéma JSON doit être rempli avec `"unit": "windows"` et une note explicite, jamais silencieusement en heures approximées.

## 11. Étape 11 — Architecture logicielle

| Catégorie | Fichiers |
|---|---|
| **Nouveaux, à créer plus tard (aucun créé maintenant)** | `Training/evaluation/models/hmm.py` (`register_model("hmm")`, `register_model("hmm_constrained")`) ; `Training/evaluation/run_hmm_evaluation.py` (mirroir `run_gru_evaluation.py`) ; `Training/evaluation/run_hmm_test_evaluation.py` (mirroir `run_gru_test_evaluation.py`) ; `Training/evaluation/run_hmm_frame_shuffle_sanity_check.py` ; `Tests/evaluation/models/test_hmm.py` ; `Training/embeddings/join_elapsed_time.py` (§2, produit `metadata_with_time.csv`) ; `Training/evaluation/reporting.py` (génération du JSON §9/10, une fois le prototype validé) |
| **Existants, réutilisés sans modification** | `Training/evaluation/model.py` (interface `Model`), `Training/evaluation/trajectory.py` (`Trajectory`, `group_into_trajectories`, `shuffle_trajectory`), `Training/evaluation/metrics.py` (fonctions existantes, réutilisées telles quelles), `Training/evaluation/seeding.py`, `Training/embeddings/dataset.py`/`cache.py` (lecture des embeddings), `Training/DataSet.py::Embryo_Transition_Dataset` (reconstruction read-only pour §1/§2, jamais modifié) |
| **Extensions additives autorisées (nouvelles fonctions, zéro suppression/modification de l'existant)** | `Training/evaluation/metrics.py` (Brier multi-classe, entropie — nouvelles fonctions à côté de `compute_metrics` existant, jamais dedans) ; `Training/evaluation/config.py` (`ExperimentConfig` déjà générique par `model_kwargs: Dict[str, Dict]`, ne nécessite probablement aucune modification — à confirmer une fois `hmm.py` écrit) |
| **Fichiers qui doivent rester strictement inchangés** | Tout `Training/` original (`DataSet.py`, `Load_data.py`, `ModelBuilder.py`, `config_args.py`, `train.py`, `train_val_test_pipline.py`, `preProcess.py`) ; `Training/embeddings/build_cache.py`/`cache.py`/`dataset.py`/`extractor.py` (déjà modifiés cette session-ci selon `git status`, mais pas par ce plan HMM — voir [[gru_bifurcation_negative_result]]) ; tous les fichiers `evaluation/models/{base_rate,identity_dynamics,persistence,linear_ssm,gru}.py` ; tout `Results/evaluation/e1_*`/`e1_gru_*` |
| **Interface `Model` à respecter** | `fit(train, val=None) -> None`, `predict(trajectories) -> List[TrajectoryPrediction]` (ordre préservé, jamais keyé par `video_name` — `evaluation/model.py:36-50`), `save(path)`, `load(path) -> Model` (classmethod) — le HMM doit produire un `TrajectoryPrediction.consistency_flag_prob` dérivé du décodage (changement d'état filtré) pour être compatible avec `metrics.compute_metrics` sans modification |
| **Sérialisation** | JSON uniquement, jamais pickle — discipline déjà établie (`docs/PROJECT_STATE.md` §8) : `π`, `A`, coefficients de régression logistique (déjà sérialisables via le même pattern que `identity_dynamics.py`/`persistence.py`/`linear_ssm.py`, à vérifier une fois ces fichiers lus en détail — non fait dans cette session) |
| **Configuration** | `ExperimentConfig` existant (`Training/evaluation/config.py`) — un nouveau `run_metadata.json` par run HMM (checksum embedding, config contraintes, seed, commit, date — cohérent avec `docs/HMM_RESEARCH_PLAN.md` §11) |
| **Reproductibilité** | `π`/`A` par comptage fermé = déterministe à 100% (pas de seed nécessaire pour cette partie) ; l'émission (régression logistique) hérite du même besoin de seed que `identity_dynamics`/`persistence` (`sklearn`, seed fixe déjà pratique standard du repo) |

---

## 12. Étape 12 — Plan d'exécution (phases, avec ce qui est concrètement bloqué)

| Phase | Objectif | Fichiers concernés | Tests | Résultat attendu | GO/NO-GO |
|---|---|---|---|---|---|
| **0** | Vérifier `phase_to_index` cross-split (§1, le risque le plus urgent) + relire un exemplaire réel de `*_timeElapsed.csv` (§2) | Aucun nouveau fichier — reconstruction read-only de 3× `Embryo_Transition_Dataset` + `cat`/`head` sur un fichier `timeElapsed.csv` réel sur le serveur | — | Confirmation ou infirmation des deux plus gros risques non résolus de ce document | **Bloquant pour tout le reste** — à faire en premier, avant toute ligne de code |
| **1** | `metadata_with_time.csv` (§2), une fois Phase 0 confirmée | `Training/embeddings/join_elapsed_time.py` (nouveau) | Test synthétique du join sur données factices (clé de jointure, cas manquant, doublon) | Table de durées réelles par fenêtre | GO conditionnel à Phase 0 |
| **2** | Analyse empirique des phases/transitions (§3) | Script d'analyse, pas encore de modèle | — | Matrices de transition (2 niveaux), histogrammes de durée, taux de retour en arrière réel | GO indépendant de Phase 1 pour la partie "niveau fenêtre" ; dépend de Phase 1 pour les durées réelles |
| **3** | HMM minimal, émission hybride (§4) | `evaluation/models/hmm.py`, `Tests/evaluation/models/test_hmm.py` | Unitaires synthétiques : `π`/`A` corrects sur séquence connue, `test_hmm_filtering_is_causal` | Smoke test CPU sur sous-ensemble réel (mirroir STEP 3 GRU) | GO indépendant de Phase 1, dépend de Phase 0 (état-espace fiable) |
| **4** | Validation du reporting probabiliste (§5, §9-10) | `evaluation/reporting.py` | Test de non-régression sur les distinctions filtrage/lissage/Viterbi | Sortie JSON exemple validée à la main sur 1-2 trajectoires | GO après Phase 3 |
| **5** | HMM contraint (§6) | Extension de `hmm.py` | Test : zéros structurels dans `A`, aucune violation dans le décodage contraint | Comparaison quantitative libre vs contraint | GO après Phase 3, dépend de la décision "professeure" sur l'irréversibilité stricte vs souple (§6) |
| **6** | Comparaison E1/GRU (§7) | `run_hmm_test_evaluation.py` | Régression : tests existants toujours verts | `REPORT.md` HMM vs 6 lignes déjà mesurées | **Autorisation explicite utilisateur requise avant tout calcul sur Test**, comme pour chaque STEP GRU |
| **7** | Analyse des durées + GO/NO-GO Semi-HMM (§8) | Script d'analyse | — | Verdict géométrique-suffisant vs Semi-HMM-justifié | Dépend strictement de Phase 1 |
| **8** | Semi-HMM, si Phase 7 = GO | `evaluation/models/semi_hmm.py` | Unitaires + comparaison durée prédite/réelle | Table durée par phase avec IC | Conditionnel |
| **9** | Reporting JSON final figé | `evaluation/reporting_schema.py` | Validation de schéma | Schéma versionné stable | — |
| **10, 11** | RAG, application web | Hors scope de ce plan (`RESEARCH_BLUEPRINT.md` Part V : "NOT NOW") | — | — | Non planifiées |

---

## 13. Étape 13 — Protection des résultats existants

Stratégie inchangée par rapport à `docs/HMM_RESEARCH_PLAN.md` §11, reconfirmée ici : structure `Results/evaluation/e1_hmm_step{N}_<étape>/` en miroir exact de la convention déjà en place pour GRU (`e1_gru_step6a_full/`, etc. — le repo n'utilise pas `Results/evaluation/HMM/` en préfixe court, il utilise `e1_<modèle>_step<N>_<nom_étape>/`, convention à respecter plutôt qu'à réinventer). Aucun script HMM n'écrit jamais dans `e1_full_analysis/`, `e1_frame_shuffle/`, `linear_ssm_dynamics_report.json`, ou `e1_gru_step6{a,b,c}_*/`. `metadata_with_time.csv` est additif à côté de `metadata.csv`, jamais en remplacement.

---

## 14. Étape 14 — Rapport final

### A. État actuel
Infrastructure d'évaluation (`Model`/`Trajectory`/`metrics`) déjà 100% réutilisable pour un HMM sans modification. Embeddings 512D en cache (Train/Val/Test connus, nombres exacts §1). Labels de phase déjà présents gratuitement dans `metadata.csv` (`first_frame_phase`/`last_frame_phase`). Aucun code HMM n'existe. Aucune donnée temporelle réelle n'est jointe. Le mapping phase→index n'est pas garanti identique entre splits — **non vérifié à ce jour**.

### B. Ce qu'il faut ajouter
Dans l'ordre de dépendance : (1) vérification `phase_to_index` cross-split + lecture d'un fichier `timeElapsed.csv` réel (Phase 0, bloquant) ; (2) script de join temporel (Phase 1) ; (3) analyse empirique des phases/transitions/durées (Phase 2) ; (4) `evaluation/models/hmm.py` (Phase 3).

### C. Premier prototype
HMM non contraint, état = phase (`chronological_phases`, indices vérifiés cohérents inter-splits), émission hybride discriminative (régression logistique multinomiale + Bayes), `π`/`A` par comptage supervisé fermé. Évalué en decoding-accuracy + log-vraisemblance sur un sous-ensemble réel, sans toucher Test.

### D. Critères de réussite
Gain significatif (bootstrap patient-level + Holm-Bonferroni) en accuracy/recall de décodage de transition par rapport à `identity_dynamics`, qui survit à un frame-shuffle adapté (interprétation inversée par rapport à GRU — voir `HMM_RESEARCH_PLAN.md` §6), et calibration (Brier/entropie) au moins équivalente — les trois ensemble.

### E. Critères d'abandon
Pas de gain de décodage par rapport à `identity_dynamics`, ou gain qui ne survit pas au frame-shuffle adapté, ou mauvaise calibration malgré un bon score de classification (signal de sur-confiance non fiable pour un reporting clinique).

### F. Passage au Semi-HMM
Uniquement si (i) le join temporel (Phase 1) est fonctionnel avec des unités réelles ET (ii) le test géométrique-vs-Gamma/log-normale (Phase 7/§8) montre un écart significatif sur au moins une phase suffisamment représentée, pré-enregistré avant de regarder le résultat.

### G. Reporting
Format texte + JSON déjà entièrement spécifiés dans `docs/HMM_RESEARCH_PLAN.md` §7-§8 — chaque champ numérique porte son type sémantique (`posterior_filtering`, `predictive_probability`, `log_likelihood_unnormalized`, ...), jamais un nombre nu.

### H. RAG
Hors scope de ce plan (`RESEARCH_BLUEPRINT.md` Part V), mais le schéma JSON §8/§9-10 est conçu dès maintenant pour que le HMM ne calcule jamais de texte — seulement des faits structurés typés, à charge du RAG de les contextualiser plus tard.

### I. Application web
Idem, hors scope — seule exigence actuelle sur le HMM : produire une entrée JSON par fenêtre déjà vue (pas seulement la dernière), toujours en filtrage causal pour tout point antérieur au dernier, avec un flag explicite `retrospective: true/false` sur tout ce qui touche à la trajectoire globale.

### J. Prochaine action UNIQUE

**Vérifier, en lecture seule sur le serveur GPU (`gpu1.labisen.isen-ouest.fr`), les deux inconnues bloquantes identifiées dans ce document, avant d'écrire la moindre ligne de code HMM :**
1. Reconstruire les trois `Embryo_Transition_Dataset` (Train/Val/Test, arguments exacts du manifest `Embeddings/resnet18/manifest.json`) et comparer leurs `.phase_to_index` — confirmer ou infirmer que l'espace d'états est cohérent entre splits (§1).
2. Lire (ex. `head -5`) un exemplaire réel de `Data/embryo_dataset_time_elapsed/{Patient_N}_timeElapsed.csv` (ou son équivalent sur la sauvegarde NAS) pour connaître ses colonnes exactes, son unité, et sa clé de jointure (§2).

Aucune autre action — ni code, ni entraînement, ni écriture de fichier — tant que ces deux points n'ont pas de réponse.
