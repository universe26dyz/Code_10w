"""Legacy myocardial-mask provenance and geometry checks."""

from .legacy_masks import audit_legacy_subject, classify_affine_relation

__all__ = ["audit_legacy_subject", "classify_affine_relation"]
