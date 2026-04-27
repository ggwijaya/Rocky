# Rocky — Claude Instructions

## Git Workflow

- **Always develop directly on the `dev` branch.** Never create feature branches.
- Before making changes, ensure you are on `dev`: `git checkout dev && git pull origin dev`
- Commit and push directly to `dev`: `git push -u origin dev`
- Do not open pull requests unless explicitly asked.
- **Do not include the Claude session URL in commit messages.**
