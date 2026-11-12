# 🤖 Starter Kit Étudiant — Systèmes Multi-Agents RAG & SLMs Locaux
*École d'Ingénieurs — 3ème Année (Cycle Ingénieur)*

Ce dépôt est le squelette de code officiel fourni pour le projet fil rouge du cours d'IA Agentique (15h).

---

## 🚀 Démarrage Rapide

### 1. Prérequis
- [Python 3.10+](https://www.python.org/)
- [`uv`](https://docs.astral.sh/uv/) (gestionnaire d'environnement ultra-rapide)
- [Ollama](https://ollama.com/) avec les modèles requis :
  ```bash
  ollama pull qwen2.5:3b
  ollama pull nomic-embed-text
  ```

### 2. Installation de l'environnement
```bash
uv sync
```

### 3. Lancer la console interactive du Superviseur
```bash
uv run python -m src.main
```

---

## 📂 Architecture du Projet

```
code_base/
├── data/syllabus/           # 23 syllabus réels de l'école (format Markdown)
├── templates/               # Starters d'agents à copier dans src/agents/
│   ├── advisor_agent/       # Modèle conseiller d'orientation (RAG + calcul ECTS)
│   └── auditor_agent/       # Modèle auditeur de prérequis (contrôle logique)
├── templates/               # Starter d'agent universel à copier dans src/agents/
│   └── agent_template/      # Modèle générique (BaseAgentPlugin, RAG, SLM, outils, tests)
├── src/
│   ├── core/                # Socle commun figé (BaseAgentPlugin, State, Ingestion, Registry)
│   ├── orchestrator/        # Nœud Superviseur LangGraph
│   ├── agents/              # ⚠️ SEUL DOSSIER OÙ VOUS TRAVAILLEZ !
│   │   └── demo_agent/      # Agent d'exemple minimal
│   └── main.py              # CLI d'orchestration
└── tests/                   # Tests d'intégration et de conformité
    ├── test_agent_compliance.py
    └── test_ingestion.py
```

---

## 🛠️ Règles du Projet & Soumission

1. **Isolation stricte** : Créez votre branche Git (`feature/agent-<votre_agent>-equipe<N>`).
2. **Copier un starter** :
2. **Copier le starter générique** :
   ```bash
   cp -r templates/advisor_agent/ src/agents/<votre_agent>/
   cp -r templates/agent_template/ src/agents/<votre_agent>/
   ```
3. **Tester localement en continu** :
   ```bash
   # Lancer vos tests unitaires
   uv run pytest src/agents/<votre_agent>/tests/

   # Vérifier le respect du contrat d'interface
   uv run pytest tests/test_agent_compliance.py

   # Qualité & typage
   uv run ruff check src/
   uv run mypy src/
   ```
4. **Ouvrir une Pull Request (PR)** : Seul votre dossier sous `src/agents/<votre_agent>/` doit être modifié !

