# coding=utf-8

from pathlib import Path

import pytest

from optimum.neuron.models.inference.backend.pretrained_model import (
    get_compiled_model_file_name,
    get_compiled_model_path,
    list_compiled_model_paths,
)


def test_get_compiled_model_file_name_is_indexed():
    assert get_compiled_model_file_name(0) == "model_0.pt"
    assert get_compiled_model_file_name(3) == "model_3.pt"


def test_get_compiled_model_path_is_indexed(tmp_path: Path):
    path = get_compiled_model_path(tmp_path, 2)
    assert path == str(tmp_path / "model_2.pt")


def test_list_compiled_model_paths_sorted_and_contiguous(tmp_path: Path):
    (tmp_path / "model_1.pt").write_text("", encoding="utf-8")
    (tmp_path / "model_0.pt").write_text("", encoding="utf-8")
    (tmp_path / "model_2.pt").write_text("", encoding="utf-8")

    paths = list_compiled_model_paths(tmp_path)
    assert paths == [
        str(tmp_path / "model_0.pt"),
        str(tmp_path / "model_1.pt"),
        str(tmp_path / "model_2.pt"),
    ]


def test_list_compiled_model_paths_raises_on_missing_index(tmp_path: Path):
    (tmp_path / "model_0.pt").write_text("", encoding="utf-8")
    (tmp_path / "model_2.pt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="expected contiguous indices"):
        list_compiled_model_paths(tmp_path)
