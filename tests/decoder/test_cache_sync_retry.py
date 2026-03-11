# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for hub cache synchronization retry logic.

These tests mock the HuggingFace Hub API and do not require Neuron hardware.
"""

from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import HfHubHTTPError
from requests import Response

from optimum.neuron.cache.hub_cache import _SYNC_MAX_RETRIES, CompileCacheHfProxy


def _make_conflict_error(status_code):
    """Create an HfHubHTTPError with the given status code."""
    response = Response()
    response.status_code = status_code
    error = HfHubHTTPError(f"{status_code} Conflict", response=response)
    return error


def _make_http_409_error():
    return _make_conflict_error(409)


def _make_http_412_error():
    return _make_conflict_error(412)


def _make_http_500_error():
    return _make_conflict_error(500)


@pytest.fixture
def mock_proxy():
    """Create a CompileCacheHfProxy with mocked dependencies."""
    with patch.object(CompileCacheHfProxy, "__init__", lambda self, *a, **kw: None):
        proxy = CompileCacheHfProxy.__new__(CompileCacheHfProxy)
        proxy.repo_id = "test-org/test-cache"
        proxy.api = MagicMock()
        proxy.default_cache = MagicMock()
        proxy.default_cache.cache_path = "/tmp/fake-cache"
        # model_info returns a mock with .sha
        proxy.api.model_info.return_value = MagicMock(sha="abc123def456")
        return proxy


def test_409_is_conflict():
    assert CompileCacheHfProxy._is_commit_conflict(_make_http_409_error())


def test_412_is_conflict():
    assert CompileCacheHfProxy._is_commit_conflict(_make_http_412_error())


def test_500_is_not_conflict():
    assert not CompileCacheHfProxy._is_commit_conflict(_make_http_500_error())


def test_no_response_is_not_conflict():
    error = HfHubHTTPError("no response")
    error.response = None
    assert not CompileCacheHfProxy._is_commit_conflict(error)


@patch("optimum.neuron.cache.hub_cache.time.sleep")
def test_succeeds_first_try(mock_sleep, mock_proxy):
    mock_proxy._upload_folder_with_retry()
    mock_proxy.api.upload_folder.assert_called_once_with(
        repo_id="test-org/test-cache",
        folder_path="/tmp/fake-cache",
        commit_message="Synchronizing local compiler cache.",
        ignore_patterns="lock",
        parent_commit="abc123def456",
    )
    mock_sleep.assert_not_called()


@patch("optimum.neuron.cache.hub_cache.time.sleep")
def test_retries_on_409_then_succeeds(mock_sleep, mock_proxy):
    mock_proxy.api.upload_folder.side_effect = [
        _make_http_409_error(),
        None,  # success on second attempt
    ]
    mock_proxy.api.model_info.side_effect = [
        MagicMock(sha="sha_v1"),
        MagicMock(sha="sha_v2"),
    ]
    mock_proxy._upload_folder_with_retry()
    assert mock_proxy.api.upload_folder.call_count == 2
    assert mock_sleep.call_count == 1


@patch("optimum.neuron.cache.hub_cache.time.sleep")
def test_retries_on_412_then_succeeds(mock_sleep, mock_proxy):
    mock_proxy.api.upload_folder.side_effect = [
        _make_http_412_error(),
        _make_http_412_error(),
        None,  # success on third attempt
    ]
    # Return different SHAs on each call to simulate HEAD advancing
    mock_proxy.api.model_info.side_effect = [
        MagicMock(sha="sha_v1"),
        MagicMock(sha="sha_v2"),
        MagicMock(sha="sha_v3"),
    ]
    mock_proxy._upload_folder_with_retry()
    assert mock_proxy.api.upload_folder.call_count == 3
    assert mock_sleep.call_count == 2
    # Verify parent_commit was refreshed each time
    calls = mock_proxy.api.upload_folder.call_args_list
    assert calls[0].kwargs["parent_commit"] == "sha_v1"
    assert calls[1].kwargs["parent_commit"] == "sha_v2"
    assert calls[2].kwargs["parent_commit"] == "sha_v3"


@patch("optimum.neuron.cache.hub_cache.time.sleep")
def test_raises_after_max_retries(mock_sleep, mock_proxy):
    mock_proxy.api.upload_folder.side_effect = _make_http_412_error()
    with pytest.raises(HfHubHTTPError):
        mock_proxy._upload_folder_with_retry()
    assert mock_proxy.api.upload_folder.call_count == _SYNC_MAX_RETRIES + 1
    assert mock_sleep.call_count == _SYNC_MAX_RETRIES


@patch("optimum.neuron.cache.hub_cache.time.sleep")
def test_non_412_error_raises_immediately(mock_sleep, mock_proxy):
    mock_proxy.api.upload_folder.side_effect = _make_http_500_error()
    with pytest.raises(HfHubHTTPError):
        mock_proxy._upload_folder_with_retry()
    assert mock_proxy.api.upload_folder.call_count == 1
    mock_sleep.assert_not_called()


@patch("optimum.neuron.cache.hub_cache.time.sleep")
def test_passes_parent_commit(mock_sleep, mock_proxy):
    mock_proxy.api.model_info.return_value = MagicMock(sha="specific_sha_123")
    mock_proxy._upload_folder_with_retry()
    mock_proxy.api.upload_folder.assert_called_once_with(
        repo_id="test-org/test-cache",
        folder_path="/tmp/fake-cache",
        commit_message="Synchronizing local compiler cache.",
        ignore_patterns="lock",
        parent_commit="specific_sha_123",
    )
