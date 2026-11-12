# 📦 Starter de Code Unique pour les Équipes Étudiantes (`templates/`)

Bienvenue dans le dossier des templates ! Ce répertoire met à votre disposition **un starter d'agent générique complet, modulaire, typé et testé**, prêt à être copié par **toutes les équipes** dans leur propre espace sous `src/agents/<nom_agent>/`, quel que soit l'archétype choisi.

---

## 📂 Contenu du Dossier

```text
templates/
├── README.md                 # Ce guide d'utilisation pas-à-pas
└── agent_template/           # Starter générique universel pour tous les archétypes
    ├── __init__.py           # Export du plugin CustomAgentPlugin
    ├── agent.py              # Contrat BaseAgentPlugin, nœud LangGraph & intégration SLM/RAG
    ├── tools.py              # Fonctions Python déterministes (Règle d'or SLM)
    └── tests/
        └── test_agent.py     # 3 tests unitaires avec Mocking (pytest < 100ms)
```

---

## 🚀 Guide Pas-à-Pas : Installer et Personnaliser Votre Agent

### Étape 1 : Choisir votre archétype et copier le template
Consultez d'abord le catalogue complet dans [`docs/CATALOGUE_AGENTS_ETUDIANTS.md`](../docs/CATALOGUE_AGENTS_ETUDIANTS.md).

Chaque binôme copie le template générique vers son dossier de travail dédié dans `src/agents/<nom_agent>/` :

```bash
# Exemple pour l'équipe advisor (Conseiller d'Orientation) :
cp -r templates/agent_template/ src/agents/advisor/

# Exemple pour l'équipe auditor (Auditeur de Prérequis) :
cp -r templates/agent_template/ src/agents/auditor/

# Exemple pour l'équipe quiz_gen (Générateur de QCMs) :
cp -r templates/agent_template/ src/agents/quiz_gen/

# Exemple pour l'équipe internship_matcher (Matcher de Stages) :
cp -r templates/agent_template/ src/agents/internship_matcher/

# Exemple pour l'équipe workload_estimator (Estimateur de Charge) :
cp -r templates/agent_template/ src/agents/workload_estimator/

# Exemple pour l'équipe exam_planner (Planificateur de Révisions) :
cp -r templates/agent_template/ src/agents/exam_planner/
```

> [!IMPORTANT]
> Ne modifiez **JAMAIS** les fichiers situés dans `templates/` directement. Travaillez exclusivement dans votre copie sous `src/agents/<nom_agent>/`.

---

### Étape 2 : Configurer l'agent dans `agent.py`
Ouvrez le fichier `src/agents/<nom_agent>/agent.py` et personnalisez les propriétés fondamentales :

1. **`name`** : L'identifiant unique en minuscules (ex: `"advisor"`, `"quiz_gen"`, `"auditor"`).
2. **`description`** : La description fournie au superviseur. Elle doit être très explicite sur les intentions déclencheuses :
   ```python
   @property
   def name(self) -> str:
       return "quiz_gen"


   @property
   def description(self) -> str:
       return (
           "Générateur d'évaluations formatives et de QCM d'entraînement. "
           "Génère des quiz à choix multiples fondés sur les compétences visées "
           "d'une UE précise et évalue les réponses de l'étudiant."
       )
   ```
3. **Filtres RAG ciblés** : Adaptez la recherche ChromaDB aux métadonnées clés de votre cas d'usage (`code_ue`, `filiere`, `semestre`, `k <= 2`).

---

### Étape 3 : Développer vos outils métiers dans `tools.py`
Les Petits Modèles de Langage (SLM 3B) peinent sur les calculs arithmétiques stricts, la logique combinatoire ou les algorithmes de graphes.

**Déléguez systématiquement ces tâches à du code Python pur dans `tools.py` :**
- Calculs d'heures et de crédits ECTS
- Algorithmes de matching de chaînes ou de mots-clés
- Parcours de graphes et détection de cycles (DFS)
- Formatage de QCM ou parsing JSON défensif

---

### Étape 4 : Valider votre travail localement
Exécutez les tests unitaires et le linter :

```bash
uv run pytest src/agents/<nom_agent>/tests/
uv run ruff check src/agents/<nom_agent>/
uv run mypy src/agents/<nom_agent>/
```
