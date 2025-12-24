# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the CLI core: `main.rs` wires together `cli`, `config`, `logging`, and `image_io`.
- `docs/` hosts extended documentation; link to or expand it when adding features.
- `config.toml` captures runtime defaults; keep it in sync with README and note overrides in `docs/`.
- Ignore `target/` artifacts; Cargo already defaults this directory to be build-only.

## Build, Test, and Development Commands
- `cargo build --locked` compiles the CLI locally; add `--release` for performance checks (~500 ms/image).
- `cargo run -- <subcommand>` exercises features without installing (e.g., `cargo run -- crop photo.jpg --output out.jpg`).
- `cargo test` executes any unit tests; add module-targeted tests as you expand functionality.
- `cargo fmt` and `cargo clippy` keep formatting and linting consistent before pushing commits.

## Coding Style & Naming Conventions
- Follow Rust 2021 conventions: four-space indentation, `snake_case` for functions/modules, `CamelCase` for structs/enums, and `SCREAMING_SNAKE_CASE` for constants.
- Keep CLI flags/config keys aligned with README terms (`--auto-aspect`, `confidence_threshold`) to avoid confusion.
- Prefer `clap` derives for argument parsing and document deviations (if any) inline with comments.

## Testing Guidelines
- Add `#[cfg(test)]` modules beside the code they cover, starting with `src/image_io.rs` and `src/config.rs`.
- Name tests descriptively (e.g., `fn crop_preserves_aspect_when_center_weighted()`).
- Run `cargo test` locally and mention the command in PR descriptions.

## Commit & Pull Request Guidelines
- Aim for conventional commits (`feat:`, `fix:`, `docs:`) that describe the change type and intent.
- Base PRs on `main` or the active feature branch, summarize changes, link issues, and list verification commands (`cargo fmt`, `cargo test`).
- Attach screenshots/log snippets only when the change affects output or UX; otherwise keep PRs text-only.

## Configuration & Security Tips
- Prefer `--config <path>` when testing custom settings; otherwise use `config.toml` or `~/.config/icarus/config.toml`.
- Let downstream automation override model/cache directories via env vars or CLI flags to keep secrets out of the repo.
