# Syllabus : Machine Learning Operations (MLOps) & Déploiement
- **Code UE** : SD_327
- **Filière** : Intelligence Artificielle & Data Science
- **Semestre** : S10
- **Crédits ECTS** : 3 ECTS
- **Volume horaire** : 90h (25h CM, 65h TP)
- **Responsable** : Dr. Simon Ouellet (simon.ouellet@univ-poitiers.fr)

## 1. Description du Cours
Cours terminal de la spécialité Science de la Donnée : industrialisation des modèles d'IA, pipelines de données versionnés (DVC), conteneurisation Docker, registre de modèles MLflow, déploiement d'API d'inférence (FastAPI), tests de robustesse, détection de dérive statistique (Data & Concept Drift via Evidently AI) et observabilité en production.

## 2. Prérequis Obligatoires
- UE préalables obligatoires : SD_321 - Machine Learning et SD_322 - Big Data
- Conteneurisation de base Docker et commandes système Linux
- Notions de CI/CD et gestion de versions Git

## 3. Compétences Visées & Acquis d'Apprentissage
- Structurer un projet de Machine Learning en code modulaire et reproductible (Poetry/uv, tests unitaires).
- Enregistrer et versionner des artéfacts et métriques d'entraînement sous MLflow.
- Encapsuler un modèle d'IA dans un conteneur sécurisé et l'exposer via une API REST documentée.
- Déployer un pipeline automatisé de monitoring de la dérive des données et du réentraînement continu.

## 4. Modalités d'Évaluation
- Projet fil rouge de pipeline MLOps complet (CI/CD GitHub Actions + API déployée) : 70%
- Soutenance technique finale d'architecture industrielle : 30%

## 5. Bibliographie & Ressources
- *Introducing MLOps*, Mark Treveil et al. (O'Reilly).
- *Machine Learning Engineering in Action*, Ben Wilson (Manning).
