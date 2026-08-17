# Coding Rules

- Python code có type hints khi hợp lý.
- Function có single responsibility.
- Không hard-code path.
- Config dùng YAML.
- Logging thay cho print trong training pipeline.
- Không lưu secret trong source.
- Không commit dataset/checkpoint lớn.
- Không duplicate training loop cho từng model.
- Model selection phải config-driven.
- Tests phải deterministic khi có thể.
- Tên file theo `naming_convention.md`.
- Production logic nằm trong `src/`, notebook chỉ exploratory.
