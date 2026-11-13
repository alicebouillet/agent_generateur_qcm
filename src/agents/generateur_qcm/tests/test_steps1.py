import os
from src.core.ingestion import get_vector_store

# Charger le vector store
vector_store = get_vector_store(reindex_if_empty=False)

# Test 1 : Vérifier que le vector_store existe
print("[OK] Vector store loaded:", vector_store)

# Test 2 : Créer le retriever (ÉTAPE 1)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
print("[OK] Retriever created:", retriever)

# Test 3 : Tester avec une requête
user_query = "Quels sont les prérequis pour ce cours?"
relevant_docs = retriever.invoke(user_query)

# Test 4 : Afficher les résultats
print(f"\n[RESULTS] Nombre de documents trouvés: {len(relevant_docs)}")
for i, doc in enumerate(relevant_docs):
    print(f"\n[Document {i+1}]")
    print(f"Code UE: {doc.metadata.get('code_ue', 'N/A')}")
    print(f"Section: {doc.metadata.get('section_title', 'N/A')}")
    print(f"Contenu: {doc.page_content[:200]}...")  # Premiers 200 caractères