#!/usr/bin/env python3
"""
🌐 Serveur MCP (Model Context Protocol) des Syllabus Académiques
Expose les données de syllabus et les outils de prérequis via le protocole MCP standardisé (stdio).

Usage:
    uv run python scripts/syllabus_mcp_server.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Forcer UTF-8 sur Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from mcp.server.fastmcp import FastMCP  # noqa: E402

# Initialisation du serveur FastMCP officiel
mcp = FastMCP(
    "Syllabus-Academic-MCP",
    instructions="Serveur MCP fournissant l'accès aux syllabus et prérequis d'ingénieurs.",
)

SYLLABUS_DIR = ROOT_DIR / "data" / "syllabus"


def _read_syllabus_files() -> dict[str, str]:
    """Charge en mémoire le contenu de tous les syllabus Markdown."""
    syllabi: dict[str, str] = {}
    if not SYLLABUS_DIR.exists():
        return syllabi

    for p in SYLLABUS_DIR.glob("*.md"):
        code_match = re.search(r"([A-Z]{2,4}_\d{3})", p.name)
        code = code_match.group(1) if code_match else p.stem
        syllabi[code] = p.read_text(encoding="utf-8")
    return syllabi


# 1. Ressource MCP : Consultation d'un syllabus par son code UE
@mcp.resource("syllabus://{code_ue}")
def get_syllabus_resource(code_ue: str) -> str:
    """Accède au texte intégral brut du syllabus Markdown pour une UE donnée."""
    code_clean = code_ue.strip().upper()
    syllabi = _read_syllabus_files()
    if code_clean not in syllabi:
        return f"Erreur : Aucun syllabus trouvé pour le code UE '{code_clean}'."
    return syllabi[code_clean]


# 2. Outil MCP : Liste de toutes les UEs disponibles
@mcp.tool()
def lister_toutes_les_ues() -> list[dict[str, str]]:
    """Retourne l'inventaire complet des UEs d'ingénieur disponibles avec leurs titres."""
    syllabi = _read_syllabus_files()
    result = []
    for code, text in syllabi.items():
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1) if title_match else "Sans titre"
        result.append({"code_ue": code, "titre": title})
    return sorted(result, key=lambda x: x["code_ue"])


# 3. Outil MCP : Extraction ciblée des prérequis d'une UE
@mcp.tool()
def extraire_prerequis_ue(code_ue: str) -> dict[str, Any]:
    """Extrait la liste exacte des prérequis obligatoires et recommandés pour une UE donnée."""
    code_clean = code_ue.strip().upper()
    syllabi = _read_syllabus_files()

    if code_clean not in syllabi:
        return {
            "code_ue": code_clean,
            "status": "not_found",
            "message": f"UE {code_clean} introuvable.",
            "description_brute": "",
            "ues_requises": [],
        }

    text = syllabi[code_clean]
    prereq_section = re.search(
        r"##\s+Prérequis\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL | re.IGNORECASE
    )
    raw_prereqs = prereq_section.group(1).strip() if prereq_section else "Aucun prérequis formel."

    # Détection des codes d'UEs cités dans les prérequis
    mentioned_codes = re.findall(r"\b[A-Z]{2,4}_\d{3}\b", raw_prereqs)

    return {
        "code_ue": code_clean,
        "status": "success",
        "description_brute": raw_prereqs,
        "ues_requises": list(set(mentioned_codes)),
    }


# 4. Prompt MCP : Modèle d'audit d'orientation pré-configuré
@mcp.prompt()
def prompt_audit_orientation(filiere_souhaitee: str, ues_validees: str) -> str:
    """Génère un prompt d'orientation universitaire standardisé pour un étudiant."""
    return f"""Tu es un conseiller pédagogique d'école d'ingénieur.
L'étudiant souhaite s'orienter vers la filière : '{filiere_souhaitee}'.
Il a actuellement validé les UEs suivantes : '{ues_validees}'.

Analyse si ses prérequis sont suffisants, identifie les cours recommandés manquants et propose un plan d'électifs cohérent.
Utilise exclusivement les informations officielles des syllabus.
"""


if __name__ == "__main__":
    # Exécute le serveur FastMCP sur l'entrée/sortie standard (stdio JSON-RPC)
    mcp.run(transport="stdio")
