---
name: debug-bisect
description: Find the cause of a failure instead of guessing
when: Something fails and the reason is not obvious
---
1. Reproduce it with the smallest command that still fails. Write that command down.
2. Read the whole error message and the line it points at before changing anything.
3. Form one hypothesis, prove or disprove it with one command or one log line.
4. Bisect when it used to work: `git log --oneline` plus `git bisect` on the smallest test.
5. Fix the cause, then run the reproduction again and the tests around it.
