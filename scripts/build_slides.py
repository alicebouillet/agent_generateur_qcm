#!/usr/bin/env python3
"""
📦 Script de compilation des diaporamas Marp (Markdown -> HTML & PDF)
Cours d'Agentique & SLMs Locaux

Usage:
    uv run python scripts/build_slides.py               # Génère tous les HTML et PDF
    uv run python scripts/build_slides.py --format pdf   # Génère uniquement les PDFs
    uv run python scripts/build_slides.py --format html  # Génère uniquement les HTMLs
    uv run python scripts/build_slides.py --session 1    # Compile uniquement la séance 1
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Forcer l'encodage UTF-8 sur la console Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Définition des répertoires
ROOT_DIR = Path(__file__).resolve().parent.parent
SLIDES_DIR = ROOT_DIR / "slides"


def check_marp_installed() -> list[str]:
    """Vérifie la présence de Marp CLI (localement ou via npx)."""
    # 1. Vérifier si marp est installé directement dans le PATH
    if shutil.which("marp"):
        return ["marp"]

    # 2. Vérifier si npx est disponible pour appeler @marp-team/marp-cli
    if shutil.which("npx"):
        return ["npx", "@marp-team/marp-cli"]

    print("❌ Erreur : Ni 'marp' ni 'npx' (Node.js) ne sont installés ou trouvés dans le PATH.")
    print("👉 Installez Node.js (https://nodejs.org) ou installez Marp CLI : npm install -g @marp-team/marp-cli")
    sys.exit(1)


def find_slide_files(session_filter: str | None = None) -> list[Path]:
    """Recherche tous les fichiers Markdown de slides dans slides/."""
    if not SLIDES_DIR.exists():
        print(f"❌ Erreur : Dossier des slides introuvable ({SLIDES_DIR})")
        sys.exit(1)

    all_md = sorted(SLIDES_DIR.glob("session_*.md"))
    if not all_md:
        print(f"⚠️ Aucun fichier session_*.md trouvé dans {SLIDES_DIR}")
        return []

    if session_filter and session_filter.lower() != "all":
        clean_filter = session_filter.lower().replace("session_", "").replace("s", "")
        # Normaliser '1' -> '01'
        if clean_filter.isdigit():
            clean_filter = f"{int(clean_filter):02d}"

        filtered = [f for f in all_md if f"session_{clean_filter}" in f.name]
        if not filtered:
            print(f"❌ Aucune slide trouvée pour le filtre de séance : '{session_filter}'")
            print(f"   Fichiers disponibles : {[f.name for f in all_md]}")
            sys.exit(1)
        return filtered

    return all_md


def compile_slides(
    files: list[Path],
    marp_cmd: list[str],
    generate_html: bool,
    generate_pdf: bool,
) -> bool:
    """Compile la liste des fichiers Markdown en HTML et/ou PDF via Marp CLI."""
    total_files = len(files)
    print(f"\n🎨 Compilation de {total_files} présentation(s) Marp...")
    targets = []
    if generate_html:
        targets.append("HTML interactif")
    if generate_pdf:
        targets.append("PDF d'impression")
    print(f"🎯 Formats cibles : {', '.join(targets)}\n")

    start_time = time.time()
    success_count = 0
    error_count = 0

    for idx, md_file in enumerate(files, 1):
        print(f"[{idx}/{total_files}] 📄 Traitement de {md_file.name} :")

        # 1. Génération HTML
        if generate_html:
            html_out = md_file.with_suffix(".html")
            cmd = marp_cmd + [
                "--no-stdin",
                str(md_file),
                "-o",
                str(html_out),
                "--allow-local-files",
            ]
            try:
                res = subprocess.run(
                    cmd,
                    shell=(sys.platform == "win32"),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                size_kb = html_out.stat().st_size / 1024
                print(f"    ✔ HTML généré : {html_out.name} ({size_kb:.1f} Ko)")
            except subprocess.CalledProcessError as e:
                print(f"    ❌ Échec génération HTML pour {md_file.name} : {e.stderr or e.stdout}")
                error_count += 1
                continue

        # 2. Génération PDF
        if generate_pdf:
            pdf_out = md_file.with_suffix(".pdf")
            cmd = marp_cmd + [
                "--no-stdin",
                "--pdf",
                str(md_file),
                "-o",
                str(pdf_out),
                "--allow-local-files",
            ]
            try:
                res = subprocess.run(
                    cmd,
                    shell=(sys.platform == "win32"),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                size_mb = pdf_out.stat().st_size / (1024 * 1024)
                print(f"    ✔ PDF généré  : {pdf_out.name} ({size_mb:.2f} Mo)")
            except subprocess.CalledProcessError as e:
                print(f"    ❌ Échec génération PDF pour {md_file.name} : {e.stderr or e.stdout}")
                error_count += 1
                continue

        success_count += 1

    elapsed = time.time() - start_time
    print("\n" + "═" * 60)
    print(f"🏁 Bilan de compilation :")
    print(f"   • Présentations traitées avec succès : {success_count} / {total_files}")
    if error_count > 0:
        print(f"   • Erreurs rencontrées                : {error_count}")
    print(f"   • Temps d'exécution                  : {elapsed:.2f} secondes")
    print(f"   • Dossier de sortie                  : {SLIDES_DIR}")
    print("═" * 60 + "\n")

    return error_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Générateur automatisé de diaporamas HTML & PDF à partir des sources Markdown Marp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--format",
        choices=["all", "html", "pdf"],
        default="all",
        help="Format de sortie à générer (défaut : all)",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="all",
        help="Numéro ou identifiant de séance à cibler (ex: 1, 01, s2, all)",
    )

    args = parser.parse_args()

    marp_cmd = check_marp_installed()
    files = find_slide_files(args.session)
    if not files:
        sys.exit(1)

    gen_html = args.format in ("all", "html")
    gen_pdf = args.format in ("all", "pdf")

    success = compile_slides(
        files=files,
        marp_cmd=marp_cmd,
        generate_html=gen_html,
        generate_pdf=gen_pdf,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
