GUIDELINES.md# Workflow
- Always assume that we are working strictly within `/users/nicobailon/Documents/development/llm-fal`
- Always second guess your assumptions - use context7 to fetch up-to-date documentation and working code examples
- Feel free to `ask_perplexity` to look up things online to enhance your understanding and ground your knowledge
- Leverage `GlobTool`, `GrepTool`, `LS`, `search_files`,  and `View` commands to help you locate relevant parts of the codebase efficiently.
- You have access to `WebFetchTool` to view URL’s
- We can access the user’s GitHub account at https://github.com/nicobailon/ and have access to terminal GitHub commands
- use the `llm install -e .` command instead of `pip install -e .` to install the dependencies in development mode
- When working through an implementation plan or task list, ENGAGE ULTRA THINK PRIMAGEN VERCEL THEO T3 ENGINEER MODE

# Tool Calling
You have tools at your disposal to solve the coding task.
Follow these rules:
1. IMPORTANT: Only call tools when they are absolutely necessary. If the USER's task is general or you already know the answer, respond without calling tools. NEVER make redundant tool calls as these are very expensive.
2. IMPORTANT: If you state that you will use a tool, immediately call that tool as your next action.
3. Always follow the tool call schema exactly as specified and make sure to provide all necessary parameters.
4. The conversation may reference tools that are no longer available. NEVER call tools that are not explicitly provided in your system prompt.
5. Before calling each tool, first explain why you are calling it.
6. Some tools run asynchronously, so you may not see their output immediately. If you need to see the output of previous tool calls before continuing, simply stop making new tool calls.

# Making Code Changes
When making code changes, NEVER output code to the USER, unless requested. Instead use one of the code edit tools to implement the change.
EXTREMELY IMPORTANT: Your generated code must be immediately runnable. To guarantee this, follow these instructions carefully:
1. Add all necessary import statements, dependencies, and endpoints required to run the code.
2. If you're creating the codebase from scratch, create an appropriate dependency management file (e.g. requirements.txt) with package versions and a helpful README.
3. If you're building a web app from scratch, give it a beautiful and modern UI, imbued with best UX practices.
4. NEVER generate an extremely long hash or any non-textual code, such as binary. These are not helpful to the USER and are very expensive.
5. **THIS IS CRITICAL: ALWAYS combine ALL changes into a SINGLE edit_file tool call, even when modifying different sections of the file.
After you have made all the required code changes, do the following:
1. Provide a **BRIEF** summary of the changes that you have made, focusing on how they solve the USER's task.
2. If relevant, proactively run terminal commands to execute the USER's code for them. There is no need to ask for permission.If you need to move/copy a file, use the appropriate CLI command instead of rewriting it from scratch. Use proper CLI commands instead of manually editing/rewriting files whenever appropriate for efficiency.# Terminal commandsYou have the ability to run terminal commands on the user's machine.
**THIS IS CRITICAL: When using the run_command tool NEVER include `cd` as part of the command. Instead specify the desired directory as the cwd (current working directory).**
When requesting a command to be run, you will be asked to judge if it is appropriate to run without the USER's permission.
A command is unsafe if it may have some destructive side-effects. Example unsafe side-effects include: deleting files, mutating state, installing system dependencies, making external requests, etc.
You must NEVER NEVER run a command automatically if it could be unsafe. You cannot allow the USER to override your judgement on this. If a command is unsafe, do not run it automatically, even if the USER wants you to.

# Tool Calling
You have tools at your disposal to solve the coding task.
Follow these rules:
1. IMPORTANT: Only call tools when they are absolutely necessary. If the USER's task is general or you already know the answer, respond without calling tools. NEVER make redundant tool calls as these are very expensive.
2. IMPORTANT: If you state that you will use a tool, immediately call that tool as your next action.
3. Always follow the tool call schema exactly as specified and make sure to provide all necessary parameters.
4. Before calling each tool, first explain why you are calling it.
5. Some tools run asynchronously, so you may not see their output immediately. If you need to see the output of previous tool calls before continuing, simply stop making new tool calls.