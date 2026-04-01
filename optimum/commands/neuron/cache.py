# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Defines the command line related to dealing with the Neuron cache."""

from argparse import ArgumentParser

from ...neuron.cache.bucket_cache import fetch_cache
from ...neuron.cache.bucket_utils import get_cache_bucket, set_cache_bucket_in_hf_home
from ...neuron.cache.cleanup import cleanup_local_cache, get_local_cache_status
from ...neuron.cache.hub_cache import select_hub_cached_entries
from ...neuron.utils.import_utils import is_package_available
from ...neuron.utils.instance import SUPPORTED_INSTANCE_TYPES
from ...utils import logging
from ..base import BaseOptimumCLICommand, CommandInfo


logger = logging.get_logger()


class CreateCacheBucketCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: ArgumentParser):
        parser.add_argument(
            "-n",
            "--name",
            type=str,
            default=None,
            help="The bucket ID (e.g. 'my-org/my-cache'). Defaults to the configured cache bucket.",
        )
        parser.add_argument(
            "--public",
            action="store_true",
            help="If set, the created bucket will be public. By default the cache bucket is private.",
        )

    def run(self):
        from ...neuron.cache.bucket_cache import _call_server

        bucket_id = self.args.name or get_cache_bucket()
        if not bucket_id:
            logger.error("No bucket ID specified and no default bucket configured.")
            return

        # Verify bucket connectivity via the server (auto-starts if needed)
        try:
            _call_server("ping")
            logger.info(f"Cache bucket server ready for: {bucket_id}")
        except Exception as e:
            logger.error(f"Failed to start bucket server: {e}")
            return

        set_cache_bucket_in_hf_home(bucket_id)
        public_or_private = "public" if self.args.public else "private"
        logger.info(f"Neuron cache bucket set to {bucket_id} [{public_or_private}].")


class SetCacheBucketCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument("name", type=str, help="The bucket ID to use as remote cache (e.g. 'my-org/my-cache').")

    def run(self):
        set_cache_bucket_in_hf_home(self.args.name)
        logger.info(f"Neuron cache bucket set locally to {self.args.name}.")


class FetchCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument(
            "model_id",
            type=str,
            help="The model_id to pre-warm cache for.",
        )
        parser.add_argument(
            "--task",
            type=str,
            default=None,
            help="The task to fetch cache for (e.g. 'text-generation').",
        )
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="The local cache directory to download to.",
        )

    def run(self):
        fetch_cache(model_id=self.args.model_id, task=self.args.task, cache_dir=self.args.cache_dir)


class LookupCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument(
            "model_id",
            type=str,
            help="The model_id to lookup cached versions for.",
        )
        parser.add_argument(
            "--task",
            type=str,
            default=None,
            help="The task to lookup cache for (e.g. 'text-generation').",
        )
        parser.add_argument(
            "--instance_type",
            type=str,
            choices=SUPPORTED_INSTANCE_TYPES,
            help=f"Only look for cached models for the specified instance type (e.g. {SUPPORTED_INSTANCE_TYPES}).",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            choices=["bfloat16", "float16"],
            help="Only look for cached models for the specified `torch.dtype`.",
        )
        parser.add_argument(
            "--tensor_parallel_size",
            type=int,
            help="Only look for cached models with the specified tensor parallel size.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            help="Only look for cached models supporting at least the specified batch size.",
        )
        parser.add_argument(
            "--sequence_length",
            type=int,
            help="Only look for cached models supporting at least the specified sequence length.",
        )

    def _list_entries(self):
        entries = select_hub_cached_entries(
            self.args.model_id,
            task=self.args.task,
            instance_type=self.args.instance_type,
            batch_size=self.args.batch_size,
            sequence_length=self.args.sequence_length,
            tensor_parallel_size=self.args.tensor_parallel_size,
            torch_dtype=self.args.dtype,
        )
        n_entries = len(entries)
        if n_entries == 0:
            print(f"No cached entries found for {self.args.model_id}.")
            return
        title = f"Cached entries for {self.args.model_id}"
        columns = ["batch size", "sequence length", "tensor parallel", "dtype", "instance type"]
        rows = []
        for entry in entries:
            rows.append(
                (
                    str(entry.get("batch_size", "?")),
                    str(entry.get("sequence_length", "?")),
                    str(entry.get("tp_degree", entry.get("tensor_parallel_size", "?"))),
                    str(entry.get("torch_dtype", entry.get("dtype", "?"))),
                    str(entry.get("target", "?")),
                )
            )

        def _sort_key(row):
            def _int_or(val):
                try:
                    return (0, int(val))
                except (ValueError, TypeError):
                    return (1, val)

            return (_int_or(row[2]), _int_or(row[0]), _int_or(row[1]), row[3])

        rows = list(set(rows))
        rows = sorted(rows, key=_sort_key)
        if is_package_available("rich", "14.1.0"):
            from rich.console import Console
            from rich.table import Table

            table = Table(title=title)
            for column in columns:
                table.add_column(column, justify="center", no_wrap=True)
            for row in rows:
                table.add_row(*row)
            Console().print(table)
        else:
            print(title)
            row_format = "{:^16}" * len(columns)
            print(row_format.format(*columns))
            for row in rows:
                print(row_format.format(*row))

    def run(self):
        self._list_entries()


class StatusCacheCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument("--cache_dir", type=str, default=None, help="The local cache directory to inspect.")

    def run(self):
        status = get_local_cache_status(cache_dir=self.args.cache_dir)
        print(status.summary())


class CleanupCacheCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument("--cache_dir", type=str, default=None, help="The local cache directory to clean up.")
        parser.add_argument(
            "--all",
            action="store_true",
            dest="remove_all",
            help="Also remove empty/incomplete entries.",
        )
        parser.add_argument(
            "--old-versions",
            action="store_true",
            help="Remove cache directories for old compiler versions.",
        )
        parser.add_argument(
            "--wipe",
            action="store_true",
            help="Remove the entire cache directory.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be removed without actually deleting anything.",
        )

    def run(self):
        result = cleanup_local_cache(
            cache_dir=self.args.cache_dir,
            remove_failed=True,
            remove_locks=True,
            remove_empty=self.args.remove_all,
            remove_old_versions=self.args.old_versions,
            wipe=self.args.wipe,
            dry_run=self.args.dry_run,
        )
        prefix = "[DRY RUN] " if self.args.dry_run else ""
        print(f"{prefix}{result.summary()}")


class CustomCacheRepoCommand(BaseOptimumCLICommand):
    SUBCOMMANDS = (
        CommandInfo(
            name="create",
            help="Create a storage bucket on the Hugging Face Hub for Neuron compilation files.",
            subcommand_class=CreateCacheBucketCommand,
        ),
        CommandInfo(
            name="set",
            help="Set the name of the Neuron cache bucket to use locally.",
            subcommand_class=SetCacheBucketCommand,
        ),
        CommandInfo(
            name="fetch",
            help="Pre-warm the local cache by downloading MODULE dirs for a model from the bucket.",
            subcommand_class=FetchCommand,
        ),
        CommandInfo(
            name="lookup",
            help="Lookup cached export configurations for the specified model id.",
            subcommand_class=LookupCommand,
        ),
        CommandInfo(
            name="status",
            help="Show local Neuron compile cache status (entry counts by state, disk usage, compiler versions).",
            subcommand_class=StatusCacheCommand,
        ),
        CommandInfo(
            name="cleanup",
            help="Clean up poisoned entries (failed compilations, stale locks) from the local Neuron compile cache.",
            subcommand_class=CleanupCacheCommand,
        ),
    )
