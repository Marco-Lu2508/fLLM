---
name: code-review
description: Review the current diff like a reviewer who has to approve it
when: Before you say a change is done
---
1. Get the diff (`review` tool or `git diff`). Read it before judging anything.
2. For each hunk ask: what breaks here? Error paths, empty input, concurrency, off-by-one, resource leaks.
3. Check that tests cover the new behaviour, not just the old one.
4. Name each finding with file and line, say how sure you are, and what you would do instead.
5. Say plainly when the change is fine — a review that invents problems is useless.
