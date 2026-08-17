# Folder Structure Contract

## Root

```text
README.md
AGENTS.md
progress_status.md
codebase_audit.md
naming_convention.md
folder_structure.md
```

## Knowledge Base

`knowledge_base/` chứa kiến thức domain/project mà AI Agent cần tham khảo.

Không đặt training output vào đây.

## AI configuration

`.ai/` chứa:

- agent behavior;
- prompts;
- rules;
- skills.

Không đặt model checkpoints vào đây.

## docs

`docs/` chứa tài liệu kỹ thuật, nghiên cứu và báo cáo.

## src

`src/` chỉ chứa source code.

```text
src/
├── data/
├── models/
├── training/
├── evaluation/
├── inference/
├── api/
└── ui/
```

## data

```text
data/
├── raw/
└── processed/
```

Dataset lớn không commit vào Git.

## artifacts

Chứa:

- metrics;
- plots;
- predictions;
- checkpoints;
- logs.

## tests

Mirror structure của source khi phù hợp.

## notebooks

Chỉ dùng exploratory analysis.

Production logic phải nằm trong `src/`.
