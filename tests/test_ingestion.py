"""
Tests unitaires pour la pipeline d'ingestion des syllabus.
"""

from src.core.ingestion import extract_syllabus_metadata, load_and_chunk_syllabus


def test_extract_syllabus_metadata():
    sample_content = """# Syllabus : Test Course
- **Code UE** : TEST_101
- **Filière** : Informatique
- **Semestre** : S7
- **Crédits ECTS** : 5 ECTS
"""
    meta = extract_syllabus_metadata(sample_content, "test.md")
    assert meta["code_ue"] == "TEST_101"
    assert meta["filiere"] == "Informatique"
    assert meta["semestre"] == "S7"
    assert meta["ects"] == 5


def test_load_and_chunk_syllabus():
    chunks = load_and_chunk_syllabus("data/syllabus")
    assert len(chunks) > 0
    # Vérifie la présence de métadonnées essentielles des UEs de l'ENSAR
    codes = {c.metadata["code_ue"] for c in chunks}
    assert "SD_321" in codes
    assert "SD_327" in codes
    assert "SSI_107" in codes
    assert "QSE_212" in codes
