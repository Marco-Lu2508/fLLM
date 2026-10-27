---
name: release-notes
description: Turn a range of commits into release notes
when: A version is about to be released
---
1. Get the commits since the last tag: `git log --oneline vLAST..HEAD`.
2. Sort them: added, changed, fixed, removed. Drop internal noise (formatting, typos in comments).
3. Write for the user of the software, not for the author of the commit.
4. One line per change, with the pull request or commit when it helps.
