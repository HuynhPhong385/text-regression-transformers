# Codebase Audit

## Purpose

Audit định kỳ để ngăn codebase bị phình to hoặc mất tính nhất quán sau nhiều vòng Vibe Coding.

## Audit checklist

### Architecture

- [ ] Folder structure đúng `folder_structure.md`
- [ ] Module có responsibility rõ ràng
- [ ] Không circular dependency
- [ ] Không duplicate architecture

### Code quality

- [ ] Naming đúng convention
- [ ] Type hints phù hợp
- [ ] Function không quá lớn
- [ ] Không dead code
- [ ] Không duplicate logic
- [ ] Không magic numbers

### ML

- [ ] Dataset split đúng
- [ ] Không data leakage
- [ ] Seed được ghi
- [ ] Config reproducible
- [ ] Checkpoint có metadata
- [ ] Test set chỉ dùng cuối cùng

### AI Agent

- [ ] Agent đọc Knowledge Base
- [ ] Agent tuân thủ rules
- [ ] Prompt không mâu thuẫn
- [ ] Skill không duplicate
- [ ] Progress được cập nhật

### Testing

- [ ] Unit tests
- [ ] Model tests
- [ ] Integration tests
- [ ] Full pipeline smoke test

### Documentation

- [ ] README đúng command
- [ ] Docs phản ánh code hiện tại
- [ ] Experiment records đầy đủ

## Audit output

Mỗi audit phải ghi:

```text
Date:
Auditor:
Commit:
Critical issues:
Major issues:
Minor issues:
Refactoring:
Tests:
```

## Rule

Không refactor hàng loạt chỉ vì "code chưa đẹp".

Ưu tiên:
1. correctness;
2. reproducibility;
3. maintainability;
4. optimization.
