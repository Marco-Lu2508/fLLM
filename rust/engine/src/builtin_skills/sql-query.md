---
name: sql-query
description: Write or review a SQL query without breaking data
when: A query needs writing or changing
---
1. Read the schema first (tables, columns, indexes, foreign keys). Never guess column names.
2. Write the SELECT and run it with LIMIT before you touch anything.
3. For UPDATE/DELETE: run the same WHERE as a SELECT and count the rows first.
4. Use parameters, never string concatenation.
5. Wrap several changes in a transaction and check the result before COMMIT.
