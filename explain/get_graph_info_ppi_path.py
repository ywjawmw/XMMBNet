#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/4/16 09:23
# @Author  : Wenjing
# @File    : get_graph_info_ppi_path.py
# @Desc    :

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

# from get_graph_info import (
#     # DEFAULT_EXCEL_PATH,
#     # DEFAULT_OMIM_PATH,
#     # PROJECT_ROOT,
#     # get_disease_graph_summary,
#     get_drug_name,
#     get_graph_info_summary,
#     get_ppi_status_dict,
# )

FOLD_ID = 2
summary_rank_id = 9  # top K 的药ID
Disease_NMAE = 'FA'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEL_PATH = PROJECT_ROOT / "case" / "Adataset" / f"{Disease_NMAE}_Tok7-graphinfo{summary_rank_id}-protein.xlsx"
DEFAULT_OMIM_PATH = (
    PROJECT_ROOT / "name_data" / "drug_data" / "Adataset" / "omim_diseases.csv"
)


def _is_empty(value: Any) -> bool:
    return value is None or (pd.isna(value) if not isinstance(value, list) else False)


def _normalize_number(value: Any) -> Any:
    if _is_empty(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value

def _load_omim_name_map(omim_csv_path: str | Path = DEFAULT_OMIM_PATH) -> dict[str, str]:
    omim_df = pd.read_csv(omim_csv_path)
    required_columns = {"MIMID", "Name"}
    missing_columns = required_columns - set(omim_df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in OMIM csv: {sorted(missing_columns)}")

    return dict(
        zip(
            omim_df["MIMID"].astype(str).str.strip(),
            omim_df["Name"].astype(str).str.strip(),
        )
    )
def _split_neighbors(value: Any) -> list[str]:
    if _is_empty(value):
        return []
    return [item.strip() for item in str(value).split("; ") if item.strip()]

def _dedupe_neighbors(neighbors: list[str]) -> list[str]:
    deduped_neighbors = []
    seen_neighbors = set()
    for neighbor in neighbors:
        if neighbor in seen_neighbors:
            continue
        seen_neighbors.add(neighbor)
        deduped_neighbors.append(neighbor)
    return deduped_neighbors

def _map_omim_neighbors(neighbors: Any, omim_name_map: dict[str, str]) -> list[str]:
    mapped_neighbors = [
        omim_name_map.get(neighbor, neighbor) for neighbor in _split_neighbors(neighbors)
    ]
    return _dedupe_neighbors(mapped_neighbors)

def _build_score_dict(
    df: pd.DataFrame,
    *,
    omim_name_map: dict[str, str],
    disease_similarity_score_name: str | None = None,
) -> dict[str, dict[str, Any]]:
    score_dict: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        score_name = str(row["score_name"]).strip()
        neighbors = _dedupe_neighbors(_split_neighbors(row["neighbors"]))

        if score_name == "evi_score":
            neighbors = []
        elif score_name == "dir_score":
            neighbors = _map_omim_neighbors(row["neighbors"], omim_name_map)
        elif score_name == disease_similarity_score_name:
            neighbors = _map_omim_neighbors(row["neighbors"], omim_name_map)

        score_dict[score_name] = {
            "score_value": _normalize_number(row["score_value"]),
            "neighbor_count": len(neighbors),
            "neighbors": neighbors,
        }

    return score_dict

def get_disease_graph_summary(
    disease_id: int = 439,
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    omim_csv_path: str | Path = DEFAULT_OMIM_PATH,
    fold_id = 1
) -> dict[str, Any]:
    """Read DiseaseGraph_Summary_F1 and return disease and score information."""

    sheet_name = f"DiseaseGraph_Summary_F{fold_id}"
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    _require_columns(
        df,
        [
            "disease_id",
            "disease_name",
            "score_name",
            "score_value",
            "edge_type",
            "neighbor_count",
            "neighbors",
        ],
        sheet_name,
    )

    selected_df = df[df["disease_id"] == disease_id]
    if selected_df.empty:
        return {
            "disease_id": disease_id,
            "disease_name": None,
            "scores": {},
        }

    omim_name_map = _load_omim_name_map(omim_csv_path)
    disease_omim_id = str(selected_df.iloc[0]["disease_name"]).strip()

    disease_similarity_rows = selected_df[
        selected_df["edge_type"].astype(str).str.strip() == "disease-disease similarity"
    ]
    disease_similarity_score_name = None
    if not disease_similarity_rows.empty:
        disease_similarity_score_name = str(
            disease_similarity_rows.iloc[0]["score_name"]
        ).strip()

    return {
        "disease_id": disease_id,
        "disease_name": omim_name_map.get(disease_omim_id, disease_omim_id),
        "scores": _build_score_dict(
            selected_df,
            omim_name_map=omim_name_map,
            disease_similarity_score_name=disease_similarity_score_name,
        ),
    }

def _resolve_summary_rank(df: pd.DataFrame, summary_rank_id: int) -> Any:
    if (df["summary_rank"] == summary_rank_id).any():
        return summary_rank_id
    if summary_rank_id == 0 and not df.empty:
        return df["summary_rank"].min()
    return summary_rank_id

def get_graph_info_summary(
    summary_rank_id: int = 0,
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    omim_csv_path: str | Path = DEFAULT_OMIM_PATH,
    fold_id=FOLD_ID
) -> dict[str, dict[str, Any]]:
    """Read GraphInfo_Summary_F1 and return score information for one rank.

    If ``summary_rank_id`` is 0 but the sheet is 1-based, the first available
    ``summary_rank`` is used so the default works with exported Excel files.
    """

    sheet_name = f"GraphInfo_Summary_F{fold_id}"
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    _require_columns(
        df,
        ["summary_rank", "score_name", "score_value", "neighbor_count", "neighbors"],
        sheet_name,
    )

    rank_value = _resolve_summary_rank(df, summary_rank_id)
    selected_df = df[df["summary_rank"] == rank_value]
    omim_name_map = _load_omim_name_map(omim_csv_path)

    return _build_score_dict(selected_df, omim_name_map=omim_name_map)

def get_drug_name(
    summary_rank_id: int = 0,
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    fold_id=FOLD_ID
) -> str | None:
    """Read GraphInfo_Summary_F1 and return the selected drug name."""

    sheet_name = f"GraphInfo_Summary_F{fold_id}"
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    _require_columns(df, ["summary_rank", "drug_name"], sheet_name)

    rank_value = _resolve_summary_rank(df, summary_rank_id)
    selected_df = df[df["summary_rank"] == rank_value]
    if selected_df.empty:
        return None

    return str(selected_df.iloc[0]["drug_name"]).strip()


def get_disease_graph_summary(
    disease_id: int = 439,
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    omim_csv_path: str | Path = DEFAULT_OMIM_PATH,
    fold_id = 1
) -> dict[str, Any]:
    """Read DiseaseGraph_Summary_F1 and return disease and score information."""

    sheet_name = f"DiseaseGraph_Summary_F{fold_id}"
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    _require_columns(
        df,
        [
            "disease_id",
            "disease_name",
            "score_name",
            "score_value",
            "edge_type",
            "neighbor_count",
            "neighbors",
        ],
        sheet_name,
    )

    selected_df = df[df["disease_id"] == disease_id]
    if selected_df.empty:
        return {
            "disease_id": disease_id,
            "disease_name": None,
            "scores": {},
        }

    omim_name_map = _load_omim_name_map(omim_csv_path)
    disease_omim_id = str(selected_df.iloc[0]["disease_name"]).strip()

    disease_similarity_rows = selected_df[
        selected_df["edge_type"].astype(str).str.strip() == "disease-disease similarity"
    ]
    disease_similarity_score_name = None
    if not disease_similarity_rows.empty:
        disease_similarity_score_name = str(
            disease_similarity_rows.iloc[0]["score_name"]
        ).strip()

    return {
        "disease_id": disease_id,
        "disease_name": omim_name_map.get(disease_omim_id, disease_omim_id),
        "scores": _build_score_dict(
            selected_df,
            omim_name_map=omim_name_map,
            disease_similarity_score_name=disease_similarity_score_name,
        ),
    }


def get_ppi_hop1_dict(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    fold_id = FOLD_ID
) -> dict[str, Any]:
    """Read PPI_Path3Hop_F1 and map proteins using the shortest available hop.

    The lookup checks hop_distance 1, then 2, then 3. If none are available,
    the returned flag marks the PPI graph as unreachable.
    """

    sheet_name = f"PPI_Path3Hop_F{fold_id}"
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    _require_columns(
        df,
        [
            "hop_distance",
            "disease_side_protein_symbol",
            "drug_side_protein_symbol",
        ],
        sheet_name,
    )

    selected_hop = None
    hop_df = pd.DataFrame()
    for hop_distance in (1, 2, 3):
        current_df = df[df["hop_distance"] == hop_distance]
        if not current_df.empty:
            selected_hop = hop_distance
            hop_df = current_df
            break

    if selected_hop is None:
        return {
            "flag": "no_ppi_reachable",
            "hop_distance": None,
            "protein_map": {},
        }

    ppi_dict: dict[str, list[str]] = {}
    for _, row in hop_df.iterrows():
        disease_protein = str(row["disease_side_protein_symbol"]).strip()
        drug_protein = str(row["drug_side_protein_symbol"]).strip()
        if not disease_protein or not drug_protein:
            continue
        ppi_dict.setdefault(disease_protein, [])
        if drug_protein not in ppi_dict[disease_protein]:
            ppi_dict[disease_protein].append(drug_protein)

    return {
        "flag": "ppi_reachable",
        "hop_distance": selected_hop,
        "protein_map": ppi_dict,
    }


def get_ppi_status_dict(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    fold_id = FOLD_ID
) -> dict[str, Any]:
    """Read PPI_Status_F1 and return the overlap protein status message."""

    sheet_name = f"PPI_Status_F{fold_id}"
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    _require_columns(df, ["message"], sheet_name)

    if df.empty:
        return {"overlap_protein": None}

    return {"overlap_protein": df.iloc[0]["message"]}



def _require_columns(df: pd.DataFrame, columns: list[str], sheet_name: str) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {sheet_name}: {missing_columns}")


def _clean_ppi_node(node: Any) -> str:
    return re.sub(r"\([^)]*\)", "", str(node)).strip()


def _parse_reversed_ppi_path(ppi_path: Any) -> list[str]:
    if pd.isna(ppi_path):
        return []
    nodes = [_clean_ppi_node(node) for node in str(ppi_path).split("->")]
    nodes = [node for node in nodes if node]
    return list(reversed(nodes))


def _join_or(items: list[str]) -> str:
    return "或".join(items)


def _join_path(items: list[str]) -> str:
    return "->".join(items)


def _join_path_alternatives(items: list[str]) -> str:
    return "，或".join(items)


def _dedupe(items: list[str]) -> list[str]:
    deduped_items = []
    seen_items = set()
    for item in items:
        if item in seen_items:
            continue
        seen_items.add(item)
        deduped_items.append(item)
    return deduped_items


def _format_hop_description(
    *,
    hop_distance: int,
    disease_name: str | None,
    drug_name: str | None,
    disease_protein: str,
    path_tails: list[list[str]],
) -> str:
    disease_display = disease_name or ""
    drug_display = drug_name or ""

    if hop_distance == 1:
        connected_proteins = _join_or(_dedupe([tail[0] for tail in path_tails if tail]))
        return (
            f"{disease_display}相关蛋白为{disease_protein}, "
            f"通过连接蛋白{connected_proteins}，连接药物{drug_display}\n"
        )

    tail_texts = [_join_path(tail) for tail in path_tails if tail]
    connected_paths = _join_path_alternatives(_dedupe(tail_texts))
    return (
        f"{disease_display}相关蛋白为{disease_protein}, "
        f"通过连接蛋白{connected_paths}，连接药物{drug_display}\n"
    )


def get_ppi_path_dict(
    drug_name: str | None,
    disease_name: str | None,
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    fold_id=FOLD_ID
) -> dict[str, Any]:
    """Read PPI_Path3Hop_F1 and return hop 1/2/3 path descriptions."""

    sheet_name = f"PPI_Path3Hop_F{fold_id}"
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    _require_columns(
        df,
        ["hop_distance", "disease_side_protein_symbol", "ppi_path"],
        sheet_name,
    )

    protein_map = {
        "hop_1": [],
        "hop_2": [],
        "hop_3": [],
    }

    for hop_distance in (1, 2, 3):
        hop_df = df[df["hop_distance"] == hop_distance]
        paths_by_disease_protein: dict[str, list[list[str]]] = {}

        for _, row in hop_df.iterrows():
            reversed_path = _parse_reversed_ppi_path(row["ppi_path"])
            if len(reversed_path) < 2:
                continue

            disease_protein = reversed_path[0]
            fallback_disease_protein = str(row["disease_side_protein_symbol"]).strip()
            if not disease_protein:
                disease_protein = fallback_disease_protein

            path_tail = reversed_path[1:]
            paths_by_disease_protein.setdefault(disease_protein, [])
            if path_tail not in paths_by_disease_protein[disease_protein]:
                paths_by_disease_protein[disease_protein].append(path_tail)

        for disease_protein, path_tails in paths_by_disease_protein.items():
            protein_map[f"hop_{hop_distance}"].append(
                _format_hop_description(
                    hop_distance=hop_distance,
                    disease_name=disease_name,
                    drug_name=drug_name,
                    disease_protein=disease_protein,
                    path_tails=path_tails,
                )
            )

    has_ppi_path = any(protein_map[hop_key] for hop_key in protein_map)
    if not has_ppi_path:
        return {
            "flag": "no_ppi_reachable",
            "protein_map": protein_map,
        }

    return {
        "flag": "ppi_reachable",
        "protein_map": protein_map,
    }


def get_graph_info_ppi_path(
    summary_rank_id: int = 0,
    disease_id: int = 439,
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    omim_csv_path: str | Path = DEFAULT_OMIM_PATH,
    fold_id: int = FOLD_ID
) -> dict[str, Any]:
    """Return graph information with hop-specific PPI path descriptions."""

    drug_name = get_drug_name(summary_rank_id=summary_rank_id, excel_path=excel_path)
    disease_graph_summary = get_disease_graph_summary(
        disease_id=disease_id,
        excel_path=excel_path,
        omim_csv_path=omim_csv_path,
        fold_id=fold_id
    )
    disease_name = disease_graph_summary["disease_name"]

    return {
        "drug_name": drug_name,
        "disease_name": disease_name,
        "graph_info_summary": get_graph_info_summary(
            summary_rank_id=summary_rank_id,
            excel_path=excel_path,
            omim_csv_path=omim_csv_path,
        ),
        "disease_graph_summary": disease_graph_summary,
        "ppi_status": get_ppi_status_dict(excel_path=excel_path),
        "ppi_hop1": get_ppi_path_dict(
            drug_name=drug_name,
            disease_name=disease_name,
            excel_path=excel_path,
        ),
    }


if __name__ == "__main__":
    disease_id = 439


    res = {
        "keyword": "graph_info",
        "example": []
    }
    result = get_graph_info_ppi_path(
        summary_rank_id=summary_rank_id,
        disease_id=disease_id,
    )

    res["example"].append(result)
    output_path = (
            PROJECT_ROOT / "explain" / "graph_information" / f"{Disease_NMAE}_graph_info_ppi_path_top{summary_rank_id}_fold{FOLD_ID}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(res, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved graph info json to: {output_path}")
