---
name: tests-first
description: Add tests that would have caught the bug
when: You fixed a bug or added behaviour
---
1. Write the failing test first: it has to fail for the reason the bug describes.
2. Run it, see it fail, then fix the code until it passes.
3. Cover the edge cases the bug touched: empty, huge, wrong type, repeated call.
4. Keep the test readable: a name that says the behaviour, no sleep, no network.
