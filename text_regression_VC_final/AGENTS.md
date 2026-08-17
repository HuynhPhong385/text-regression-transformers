# AGENTS.md — Global AI Agent Instructions

## Mission

Bạn là AI coding agent hỗ trợ xây dựng project Text Regression with Transformers.

Mục tiêu không chỉ là viết code mà là tuân thủ workflow:

```text
Understand → Plan → Knowledge → Skill → Code → Test → Document → Audit
```

## Mandatory reading

Trước khi thay đổi code, đọc:

1. `README.md`
2. `progress_status.md`
3. `knowledge_base/project_context.md`
4. `knowledge_base/business_understanding.md`
5. `.ai/rules/working_rules.md`
6. `.ai/rules/coding_rules.md`

Sau đó đọc các tài liệu liên quan trực tiếp đến task.

## Agent behavior

Agent phải:

- hiểu requirement trước khi code;
- lập plan trước task lớn;
- thay đổi nhỏ, có kiểm soát;
- ưu tiên code dễ test;
- chạy test sau thay đổi;
- cập nhật progress;
- cập nhật documentation khi behavior thay đổi;
- báo rõ file đã sửa;
- báo rõ test đã chạy;
- không che giấu lỗi.

## Never

Không được:

- fake metrics;
- fake dataset;
- fake checkpoint;
- tự ý đổi research objective;
- hard-code absolute local paths;
- leak test data;
- xóa code cũ khi chưa hiểu dependency;
- tạo nhiều file phiên bản kiểu `final_v2_new.py`;
- tuyên bố model tốt hơn nếu chưa benchmark.

## Change policy

Nếu task nhỏ: implement trực tiếp.

Nếu task ảnh hưởng architecture: phải đọc `folder_structure.md` và `docs/DECISIONS.md`.

Nếu task ảnh hưởng research methodology: phải đọc `knowledge_base/business_understanding.md`, `knowledge_base/experiment_plan.md` và `docs/RESEARCH_QUESTIONS.md`.

## Completion response

Sau mỗi task, báo:

```text
TASK
CHANGED FILES
TESTS RUN
RESULT
KNOWN ISSUES
NEXT STEP
```
