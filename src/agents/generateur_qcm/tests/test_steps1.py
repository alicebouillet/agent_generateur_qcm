"""Test ÉTAPE 1 du Générateur de QCM: RAG ciblé avec filtres sur code_ue."""
"""Commande pour lancer ce test : python -m src.agents.generateur_qcm.tests.test_steps1"""
import os
from src.core.ingestion import get_vector_store

# Charger le vector store
vector_store = get_vector_store(reindex_if_empty=False)

# Test 1 : Vérifier que le vector_store existe
print("[OK] Vector store loaded:", vector_store)

# Test 2 : Créer le retriever (ÉTAPE 1 - spécifique au QCM)
# Le générateur de QCM doit chercher un UE spécifique
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
print("[OK] Retriever created:", retriever)

# Test 3 : Simuler une requête du générateur de QCM
# Exemple: "Génère un QCM sur les objectifs d'apprentissage de SD_106"
user_query = "Genere un QCM sur les objectifs pedagogiques et contenus de SD_106"
relevant_docs = retriever.invoke(user_query)

# Test 4 : Formater le contexte RAG (comme dans l'agent)
print(f"\n[RESULTS] Nombre de documents trouvés: {len(relevant_docs)}")

context_text = ""
try:
    context_text = "\n\n".join(
        f"[Extrait {doc.metadata.get('code_ue', 'UE')} - {doc.metadata.get('section_title', '')}]\n{doc.page_content}"
        for doc in relevant_docs
    )
except Exception as e:
    context_text = f"Information non disponible (Erreur RAG: {e})"

print(f"\n[CONTEXTE RAG FORMATTE]:\n{context_text[:500]}...")  # Affiche les 500 premiers caractères

# Test 5 : Vérifier que les documents contiennent les infos pour générer un QCM
if relevant_docs:
    print("\n[VALIDATION] Documents trouvés contiennent:")
    for i, doc in enumerate(relevant_docs):
        has_objectives = "objectif" in doc.page_content.lower() or "competence" in doc.page_content.lower()
        has_content = len(doc.page_content) > 100
        print(f"  Doc {i+1}: UE={doc.metadata.get('code_ue')}, Section={doc.metadata.get('section_title')}")
        print(f"    - Contient des objectifs/competences: {has_objectives}")
        print(f"    - Contenu suffisant (>100 chars): {has_content}")
else:
    print("\n[ERREUR] Aucun document trouve - Le QCM ne peut pas etre genere!")