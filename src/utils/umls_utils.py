def get_umls_relation_type(abbreviation: str):
    relation_type_mapping = {
        "AQ": "allowed_qualifier",
        "CHD": "has_child",
        "DEL": "deleted_concept",
        "PAR": "has_parent",
        "QB": "can_be_qualified_by",
        "RB": "has_broader",
        "RL": "is_similar_to",
        "RN": "has_narrower",
        "RO": "has_other_relationship",
        "RQ": "related_possibly_synonymous",
        "RU": "related_unspecified",
        "SY": "source_asserted_synonymy",
        "XR": "not_related_no_mapping"
    }

    return relation_type_mapping.get(abbreviation.upper(), abbreviation)
