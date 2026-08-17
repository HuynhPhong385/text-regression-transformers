# Test Strategy

## Test pyramid

```text
             Full Pipeline
                  ▲
             Integration
                  ▲
              Model Tests
                  ▲
              Unit Tests
```

## 1. Unit

Test:
- target conversion;
- data validation;
- metric functions;
- config parsing.

## 2. Model level

Test:
- forward shape;
- trainable parameters;
- frozen behavior;
- checkpoint reload;
- deterministic inference when configured.

## 3. Integration

Test:
- dataset → tokenizer;
- tokenizer → model;
- model → evaluator;
- checkpoint → inference.

## 4. Full pipeline

Test realistic flow:

```text
review
→ API
→ tokenizer
→ model
→ score
→ response
```

## Performance acceptance

Record actual:
- latency;
- throughput where relevant;
- memory;
- training duration.

Không đặt threshold tùy tiện nếu chưa có project requirement.
