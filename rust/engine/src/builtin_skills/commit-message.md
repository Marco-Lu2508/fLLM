---
name: commit-message
description: Write a commit message from the staged diff
when: You are about to commit and need a message
---
1. Run `git diff --staged` (or `git diff` when nothing is staged) and read the whole diff.
2. Group the change into one subject line under 72 characters, imperative mood, no trailing period.
3. Add a body only when the reason is not obvious from the diff: what was wrong, why this way.
4. Never write "update", "fix stuff" or a list of files. Name the behaviour that changed.
