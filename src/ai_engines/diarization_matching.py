"""Thuat toan anh xa danh tinh nguoi noi Hungarian va ty le bao phu."""

from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_speaker_mapping(
    pyannote_tracks: List[Dict[str, Any]],
    campp_clusters: Dict[str, Dict[str, Any]],
    cues: List[Dict[str, Any]],
) -> Tuple[Dict[str, str], List[str]]:
    """Anh xa nhan pyannote sang nhan CAM++ bang thuat toan Hungarian voi chi phi ket hop.

    Tham so:
        pyannote_tracks: Danh sach doan thoi gian phan doan tu pyannote.
        campp_clusters: Thong tin cum gom chi so phan doan va centroid tu CAM++.
        cues: Danh sach cac phan doan kem vector dac trung.

    Tra ve:
        Bo gom tu dien anh xa nhan va danh sach nhan phat sinh moi.
    """
    pyannote_speakers = sorted(list({t["speaker"] for t in pyannote_tracks if "speaker" in t}))
    campp_speakers = sorted(list(campp_clusters.keys()))

    M = len(pyannote_speakers)
    K = len(campp_speakers)

    if M == 0 or K == 0:
        return {}, pyannote_speakers

    cost_matrix = np.zeros((M, K), dtype=np.float64)

    # Tinh toan vector dac trung trung binh cho tung nhan pyannote
    pyannote_vectors = {}
    for i, p_spk in enumerate(pyannote_speakers):
        p_tracks = [t for t in pyannote_tracks if t.get("speaker") == p_spk]
        matching_embeddings = []
        for cue in cues:
            cue_dur = max(1e-6, cue["end"] - cue["start"])
            overlap = 0.0
            for pt in p_tracks:
                st = max(cue["start"], pt["start"])
                en = min(cue["end"], pt["end"])
                if en > st:
                    overlap += (en - st)
            if (overlap / cue_dur) >= 0.5 and "embedding" in cue and cue["embedding"] is not None:
                matching_embeddings.append(cue["embedding"])
        if matching_embeddings:
            pyannote_vectors[p_spk] = np.mean(matching_embeddings, axis=0)
        else:
            pyannote_vectors[p_spk] = None

    # Xay dung ma tran chi phi dua tren trong so thoi gian va tuong dong cosin
    for i, p_spk in enumerate(pyannote_speakers):
        p_tracks = [t for t in pyannote_tracks if t.get("speaker") == p_spk]
        for j, c_spk in enumerate(campp_speakers):
            cluster_info = campp_clusters[c_spk]
            cue_ids = cluster_info.get("cue_indices", [])
            relevant_cues = [c for c in cues if c.get("id") in cue_ids]

            # Tuong dong thoi gian trung khop
            total_cue_dur = sum(max(1e-6, c["end"] - c["start"]) for c in relevant_cues)
            total_overlap = 0.0
            for c in relevant_cues:
                for pt in p_tracks:
                    st = max(c["start"], pt["start"])
                    en = min(c["end"], pt["end"])
                    if en > st:
                        total_overlap += (en - st)
            s_overlap = min(1.0, total_overlap / (total_cue_dur + 1e-6))

            # Tuong dong cosin giua cac vector dac trung
            p_vec = pyannote_vectors[p_spk]
            c_vec = cluster_info.get("centroid")
            if p_vec is not None and c_vec is not None:
                norm_p = np.linalg.norm(p_vec)
                norm_c = np.linalg.norm(c_vec)
                if norm_p > 1e-6 and norm_c > 1e-6:
                    cos = np.dot(p_vec, c_vec) / (norm_p * norm_c)
                    s_cosine = float((cos + 1.0) / 2.0)
                else:
                    s_cosine = 0.5
            else:
                s_cosine = 0.5

            similarity = 0.6 * s_overlap + 0.4 * s_cosine
            cost_matrix[i, j] = 1.0 - similarity

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[pyannote_speakers[r]] = campp_speakers[c]

    # Cac nhan phat sinh bo sung khi so nguoi noi pyannote vuot qua CAM++
    new_speakers = [pyannote_speakers[i] for i in range(M) if i not in row_ind]
    return mapping, new_speakers


def assign_cues_with_coverage(
    cues: List[Dict[str, Any]],
    pyannote_tracks: List[Dict[str, Any]],
    mapping: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Gan nhan nguoi noi cho cac cau phu de dua tren ty le bao phu thoi gian.

    Tham so:
        cues: Danh sach cac phan doan phu de.
        pyannote_tracks: Danh sach phan doan pyannote.
        mapping: Tu dien anh xa nhan pyannote sang CAM++.

    Tra ve:
        Danh sach phan doan da duoc gan nhan va danh dau co kiem tra.
    """
    results = []
    for cue in cues:
        cue_res = dict(cue)
        cue_dur = max(1e-6, cue["end"] - cue["start"])
        spk_overlaps: Dict[str, float] = {}

        for pt in pyannote_tracks:
            st = max(cue["start"], pt["start"])
            en = min(cue["end"], pt["end"])
            if en > st:
                spk = pt.get("speaker", "")
                spk_overlaps[spk] = spk_overlaps.get(spk, 0.0) + (en - st)

        sorted_spks = sorted(spk_overlaps.items(), key=lambda x: x[1], reverse=True)
        if sorted_spks:
            top_spk, top_dur = sorted_spks[0]
            r_max = top_dur / cue_dur
            if r_max >= 0.5:
                assigned = mapping.get(top_spk, top_spk)
                cue_res["assigned_speaker"] = assigned
                # Kiem tra noi chong khi khoang chenh lech thoi gian duoi 10 phan tram
                overlap_flag = False
                if len(sorted_spks) > 1:
                    sec_spk, sec_dur = sorted_spks[1]
                    if ((top_dur - sec_dur) / cue_dur) < 0.10:
                        overlap_flag = True
                cue_res["overlap_detected"] = overlap_flag
                cue_res["review_needed"] = overlap_flag or (assigned != cue.get("campp_speaker"))
            else:
                cue_res["assigned_speaker"] = cue.get("campp_speaker")
                cue_res["review_needed"] = True
                cue_res["overlap_detected"] = False
        else:
            cue_res["assigned_speaker"] = cue.get("campp_speaker")
            cue_res["review_needed"] = True
            cue_res["overlap_detected"] = False

        results.append(cue_res)
    return results


def evaluate_binary_quality_gate(
    pyannote_count: int,
    campp_count: int,
    frame_agreement: float,
    exact_speakers: Any = None,
) -> bool:
    """Danh gia cong chat luong nhi phan de quyet dinh chap nhan hoac quay ve ket qua goc.

    Tham so:
        pyannote_count: So luong nguoi noi uoc tinh tu pyannote.
        campp_count: So luong nguoi noi uoc tinh tu CAM++.
        frame_agreement: Ty le dong thuan khung hinh giua hai he thong.
        exact_speakers: So nguoi noi chinh xac do nguoi dung chi dinh neu co.

    Tra ve:
        Gia tri True neu dat chuan chat luong, nguoc lai False.
    """
    if exact_speakers is None:
        if abs(pyannote_count - campp_count) > 2:
            return False
    if frame_agreement < 0.35:
        return False
    return True
