"""Stable external JSON contract IDs with backward-compatible local file reads.

New exporters use unnumbered IDs. Readers may still accept locally archived
contract IDs containing a historical numeric release segment.
"""
from __future__ import annotations
import re


def normalize_contract(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return ''
    contract = value.strip().lower()
    if contract.startswith('neuroedge-'):
        contract = contract[len('neuroedge-'):]
    contract = re.sub(r'^r\d+(?:\.\d+)?-', '', contract)
    for prefix in ('existing-', 'prior implementation-'):
        if contract.startswith(prefix):
            contract = contract[len(prefix):]
    return 'neuroedge-' + contract
