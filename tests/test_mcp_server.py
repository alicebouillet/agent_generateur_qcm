"""
Tests unitaires pour le serveur FastMCP des syllabus académiques.
Séance Capsule : Model Context Protocol (MCP).
"""

from scripts.syllabus_mcp_server import (
    extraire_prerequis_ue,
    get_syllabus_resource,
    lister_toutes_les_ues,
    mcp,
    prompt_audit_orientation,
)


def test_mcp_instance_and_metadata():
    """Vérifie que l'instance FastMCP est correctement configurée."""
    assert mcp.name == "Syllabus-Academic-MCP"
    assert "syllabus" in mcp.instructions.lower()


def test_lister_toutes_les_ues():
    """Vérifie que l'outil liste l'ensemble des UEs disponibles."""
    ues = lister_toutes_les_ues()
    assert isinstance(ues, list)
    assert len(ues) > 0

    codes = [item["code_ue"] for item in ues]
    # Doit contenir les UEs clés du programme
    assert "SD_321" in codes
    assert "SSI_107" in codes

    # Vérification de la structure
    for item in ues:
        assert "code_ue" in item
        assert "titre" in item
        assert len(item["titre"]) > 0


def test_get_syllabus_resource_found():
    """Vérifie l'accès à une ressource existante via l'URI syllabus://."""
    content = get_syllabus_resource("SD_321")
    assert isinstance(content, str)
    assert "SD_321" in content or "Machine Learning" in content
    assert "Prérequis" in content or "Description" in content


def test_get_syllabus_resource_not_found():
    """Vérifie le message d'erreur quand une ressource est inconnue."""
    content = get_syllabus_resource("INCONNU_999")
    assert "Erreur" in content or "introuvable" in content.lower()


def test_extraire_prerequis_ue_valid():
    """Vérifie l'extraction ciblée des prérequis pour une UE réelle."""
    result = extraire_prerequis_ue("SD_321")
    assert result["status"] == "success"
    assert result["code_ue"] == "SD_321"
    assert "description_brute" in result
    assert isinstance(result["ues_requises"], list)


def test_extraire_prerequis_ue_not_found():
    """Vérifie le comportement de l'outil pour une UE inexistante."""
    result = extraire_prerequis_ue("UNKNOWN_404")
    assert result["status"] == "not_found"
    assert result["code_ue"] == "UNKNOWN_404"
    assert result["ues_requises"] == []


def test_prompt_audit_orientation():
    """Vérifie la génération du prompt d'orientation MCP."""
    prompt_text = prompt_audit_orientation(
        filiere_souhaitee="Intelligence Artificielle",
        ues_validees="SD_217, TC_101",
    )
    assert isinstance(prompt_text, str)
    assert "Intelligence Artificielle" in prompt_text
    assert "SD_217, TC_101" in prompt_text
    assert "prérequis" in prompt_text.lower()
