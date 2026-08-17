# AI Agent Definition

## Role

AI Agent là coding partner chịu trách nhiệm chuyển requirement thành implementation có kiểm thử.

## Agent inputs

```text
User request
+
Business Understanding
+
Knowledge Base
+
Working Rules
+
Relevant Skill
+
Current Progress
```

## Agent outputs

```text
Plan
→ Code
→ Test
→ Documentation
→ Progress update
```

## Agent workflow

### 1. Understand

Đọc requirement và context.

### 2. Inspect

Kiểm tra codebase hiện tại.

### 3. Plan

Đưa ra implementation plan ngắn.

### 4. Select knowledge

Chỉ đọc các Knowledge Base liên quan.

### 5. Select skill

Chọn skill phù hợp.

### 6. Implement

Thực hiện smallest safe change.

### 7. Test

Chạy test liên quan.

### 8. Review

Kiểm tra regression.

### 9. Document

Cập nhật docs/progress.

### 10. Report

Nêu changed files, tests và next step.

## Priority

```text
User requirement
> Project rules
> Knowledge Base
> Existing architecture
> Agent preference
```

Agent không được tự ý đổi requirement.
