# Project rules

- Read `context.txt` before you change the experiment design.
- The lead agent delegates code changes to other agents. The lead reviews all code before it runs a paid experiment or reports completion.
- Keep the code small. Add a dependency or abstraction only when the current task needs it.
- Use short sentences, common words, and clear names. Add comments to explain a constraint or reason, not to repeat the code.
- Use WorkArena only for evaluation. Do not train models or tune routers on its outcomes. Do not call task cheat methods.
- Keep the prompt, observations, actions, and limits fixed for model comparisons. Save task IDs, seeds, versions, and run settings.
- Separate task failure from setup, API, browser, and budget errors. An error is not evidence that a model failed the task.
- Keep keys, managed instance details, and raw traces out of Git. Do not print credentials.
- Run offline tests before live checks. Live model calls cost money; stay within the user's approved scope and budget.
- The user has authorized regular commits and pushes to the existing GitHub repository. Push small, coherent changes after lead review and relevant checks. Never include secrets or raw traces.
- Publish a results report or choose a project license only when the user explicitly asks.
