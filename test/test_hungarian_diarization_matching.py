"""Kiem thu thuat toan anh xa Hungarian va ty le bao phu phan doan nguoi noi."""

import numpy as np
from src.ai_engines.diarization_matching import (
    hungarian_speaker_mapping,
    assign_cues_with_coverage,
    evaluate_binary_quality_gate,
)


def test_hungarian_speaker_mapping_equal_counts():
    """Kiem tra anh xa khi so nguoi pyannote va CAM++ bang nhau."""
    pyannote_tracks = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0},
        {"speaker": "SPEAKER_01", "start": 10.0, "end": 20.0},
    ]
    campp_clusters = {
        "spk_0": {"cue_indices": [0], "centroid": np.array([1.0, 0.0])},
        "spk_1": {"cue_indices": [1], "centroid": np.array([0.0, 1.0])},
    }
    cues = [
        {"id": 0, "start": 0.0, "end": 10.0, "embedding": np.array([1.0, 0.0]), "campp_speaker": "spk_0"},
        {"id": 1, "start": 10.0, "end": 20.0, "embedding": np.array([0.0, 1.0]), "campp_speaker": "spk_1"},
    ]
    mapping, new_speakers = hungarian_speaker_mapping(pyannote_tracks, campp_clusters, cues)
    assert mapping["SPEAKER_00"] == "spk_0"
    assert mapping["SPEAKER_01"] == "spk_1"
    assert len(new_speakers) == 0


def test_hungarian_speaker_mapping_more_pyannote_speakers():
    """Kiem tra truong hop pyannote phat hien nhieu nguoi noi hon CAM++."""
    pyannote_tracks = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0},
        {"speaker": "SPEAKER_01", "start": 10.0, "end": 20.0},
        {"speaker": "SPEAKER_02", "start": 20.0, "end": 30.0},
    ]
    campp_clusters = {
        "spk_0": {"cue_indices": [0], "centroid": np.array([1.0, 0.0])},
        "spk_1": {"cue_indices": [1], "centroid": np.array([0.0, 1.0])},
    }
    cues = [
        {"id": 0, "start": 0.0, "end": 10.0, "embedding": np.array([1.0, 0.0]), "campp_speaker": "spk_0"},
        {"id": 1, "start": 10.0, "end": 20.0, "embedding": np.array([0.0, 1.0]), "campp_speaker": "spk_1"},
    ]
    mapping, new_speakers = hungarian_speaker_mapping(pyannote_tracks, campp_clusters, cues)
    assert len(mapping) == 2
    assert len(new_speakers) == 1
    assert "SPEAKER_02" in new_speakers


def test_hungarian_speaker_mapping_empty():
    """Kiem tra truong hop dau vao rong."""
    mapping, new_speakers = hungarian_speaker_mapping([], {}, [])
    assert mapping == {}
    assert new_speakers == []


def test_assign_cues_coverage_branching():
    """Kiem tra nhanh ty le bao phu phan doan."""
    # Phan doan bao phu hoan toan
    cues = [{"id": 0, "start": 1.0, "end": 5.0, "campp_speaker": "spk_0"}]
    pyannote_tracks = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 6.0}]
    mapping = {"SPEAKER_00": "spk_0"}
    assigned = assign_cues_with_coverage(cues, pyannote_tracks, mapping)
    assert assigned[0]["assigned_speaker"] == "spk_0"
    assert assigned[0]["review_needed"] is False
    assert assigned[0]["overlap_detected"] is False

    # Phan doan bao phu duoi 0.5 (khoang lang)
    cues_silent = [{"id": 1, "start": 10.0, "end": 20.0, "campp_speaker": "spk_0"}]
    pyannote_tracks_weak = [{"speaker": "SPEAKER_00", "start": 10.0, "end": 12.0}]  # chi bao phu 2/10 = 0.2
    assigned_silent = assign_cues_with_coverage(cues_silent, pyannote_tracks_weak, mapping)
    assert assigned_silent[0]["assigned_speaker"] == "spk_0"
    assert assigned_silent[0]["review_needed"] is True
    assert assigned_silent[0]["overlap_detected"] is False


def test_assign_cues_overlap_detection():
    """Kiem tra phat hien noi chong giua hai nguoi noi tren mot phan doan."""
    cues = [{"id": 0, "start": 0.0, "end": 10.0, "campp_speaker": "spk_0"}]
    # SPEAKER_00 chiem 5.2s, SPEAKER_01 chiem 4.8s -> chenh lech 0.4s / 10s = 0.04 < 0.10
    pyannote_tracks = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.2},
        {"speaker": "SPEAKER_01", "start": 5.2, "end": 10.0},
    ]
    mapping = {"SPEAKER_00": "spk_0", "SPEAKER_01": "spk_1"}
    assigned = assign_cues_with_coverage(cues, pyannote_tracks, mapping)
    assert assigned[0]["assigned_speaker"] == "spk_0"
    assert assigned[0]["overlap_detected"] is True
    assert assigned[0]["review_needed"] is True


def test_binary_quality_gate():
    """Kiem tra cong chat luong nhi phan."""
    # Do dong thuan cao -> dat
    assert evaluate_binary_quality_gate(pyannote_count=4, campp_count=4, frame_agreement=0.85) is True
    # Lech qua 2 nguoi -> tu choi
    assert evaluate_binary_quality_gate(pyannote_count=7, campp_count=4, frame_agreement=0.80) is False
    # Do dong thuan duoi 0.35 -> tu choi
    assert evaluate_binary_quality_gate(pyannote_count=4, campp_count=4, frame_agreement=0.25) is False
    # Khi nguoi dung da chi dinh so nguoi noi chinh xac thi bo qua kiem tra lech nguoi
    assert evaluate_binary_quality_gate(pyannote_count=7, campp_count=4, frame_agreement=0.80, exact_speakers=5) is True
