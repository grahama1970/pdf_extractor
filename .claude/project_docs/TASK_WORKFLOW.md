# TASK WORKFLOW
> Structured approach for task implementation and tracking

## Task Implementation Process
1. **Focus on one sub-task at a time**
   - Complete current sub-task before starting next
   - Get explicit user approval before proceeding

2. **Task Progress Tracking**
   - Mark completed sub-tasks: [ ] → [x]
   - Mark parent task as complete when all sub-tasks are done
   - Update "Relevant Files" section with any modified PDF processing components

3. **Implementation Sequence**
   - Verify PDF extraction requirements and context
   - Implement according to CODE_STANDARDS
   - Test with sample PDFs from src/input directory
   - Update task list and wait for approval

## Version Control Workflow
1. **Git Commits**
   - After each successfully completed task, commit the changes
   - Use descriptive commit messages with task ID reference
   - Example: `git commit -m "feat(pdf-extract): Implement schema validation for tables (Task 4.1)"`

2. **Git Tags**
   - After completing each phase, create a version tag
   - Use semantic versioning (major.minor.patch)
   - Include brief description of the phase completed
   - Example: `git tag -a v0.2.0 -m "Phase 2: Table extraction implementation complete"`

3. **Commit & Tag Verification**
   - Verify that all tests pass before committing
   - Ensure validation requirements are met before creating a tag
   - Include version number in relevant documentation

## Tools & Resources
- Second-guess assumptions about PDF structure
- Use search tools for PDF extraction techniques
- Utilize code exploration tools to understand existing extractors
- Reference existing parsers before implementing new extraction logic
