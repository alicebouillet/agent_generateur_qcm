# 🛠️ Scripts Utilitaires pour les Étudiants

Ce dossier regroupe les scripts que les équipes d'étudiants peuvent exécuter pour valider, tester et stress-tester leurs agents :

- **`crash_test.py`** : Banc d'essai automatisé simulant 10 requêtes ciblées et transversales pour évaluer la précision du routage par le Superviseur, le respect des schémas d'outils et l'absence d'erreurs non interceptées.
  ```bash
  uv run python student_scripts/crash_test.py
  ```

