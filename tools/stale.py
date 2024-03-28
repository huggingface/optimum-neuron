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
"""
Script to close stale issue. Taken from Transformers
https://github.com/huggingface/transformers/blob/main/scripts/stale.py
"""
import os
from datetime import datetime as dt

import github.GithubException
from github import Github

# TODO: define optimum specific labels
LABELS_TO_EXEMPT = [
    "bug",
    "feature request",
    "new model",
    "wip",
]


def main():
    g = Github(os.environ["COMMENT_BOT_TOKEN"])
    repo = g.get_repo("huggingface/optimum-neuron")
    open_issues = repo.get_issues(state="open")

    for i, issue in enumerate(open_issues):
        print(i, issue)
        comments = sorted(list(issue.get_comments()), key=lambda i: i.created_at, reverse=True)
        last_comment = comments[0] if len(comments) > 0 else None
        if (
            last_comment is not None
            and last_comment.user.login == "github-actions[bot]"
            and (dt.utcnow() - issue.updated_at.replace(tzinfo=None)).days > 7
            and (dt.utcnow() - issue.created_at.replace(tzinfo=None)).days >= 30
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            # print(f"Would close issue {issue.number} since it has been 7 days of inactivity since bot mention.")
            try:
                issue.edit(state="closed")
            except github.GithubException as e:
                print("Couldn't close the issue:", repr(e))
        elif (
            (dt.utcnow() - issue.updated_at.replace(tzinfo=None)).days > 23
            and (dt.utcnow() - issue.created_at.replace(tzinfo=None)).days >= 30
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            # print(f"Would add stale comment to {issue.number}")
            try:
                issue.create_comment(
                    "This issue has been automatically marked as stale because it has not had "
                    "recent activity. If you think this still needs to be addressed "
                    "please comment on this thread. Thank you!"
                )
            except github.GithubException as e:
                print("Couldn't create comment:", repr(e))


if __name__ == "__main__":
    main()
