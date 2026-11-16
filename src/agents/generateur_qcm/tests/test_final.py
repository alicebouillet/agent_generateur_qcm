"""Test ÉTAPE 4 du Générateur de QCM : génération complète de bout en bout.
Commande : python -m src.agents.generateur_qcm.tests.test_steps4
"""
from langchain_core.messages import HumanMessage

from src.agents.generateur_qcm.agent import CustomAgentPlugin

node = CustomAgentPlugin().create_node()

requetes = [
    "Génère 3 questions sur SD_106",  # cas normal
    "Fais un quiz sur TC_999",        # UE inexistante -> no_content, sans appel au SLM
]

for requete in requetes:
    print("=" * 70)
    print("DEMANDE :", requete)
    result = node({"messages": [HumanMessage(content=requete)]})
    print(result["messages"][0].content)
