# Evaluation Audit Checklist

> Phụ trách: Member 4 (Evaluation + Experiments + Error Analysis)
> Nguồn: derived from `codebase_audit.md` (mục ML + Testing) + `.ai/rules/research_rules.md`.
> Cách dùng: khi Member 2/3/5 có code, chạy từng mục và ghi kết quả vào phần "Audit log" cuối file.
> Rule: chỉ ghi PASS/FAIL kèm bằng chứng (file + dòng), không phỏng đoán.

## 1. Audit `src/data/` (Member 2) — Data integrity

| # | Mục kiểm tra | Tiêu chí PASS | Lệnh/Cách kiểm |
|---|---|---|---|
| D1 | Canonical schema | Mọi DataFrame có đúng cột `id, text, rating, target, split` | Đọc `src/data/prepare.py`, chạy smoke test in `df.columns` |
| D2 | Target conversion | `target == (rating - 1) / 4` với mọi dòng; rating trong 1–5; target trong [0,1] | `assert ((df.rating - 1) / 4 == df.target).all()` |
| D3 | Split đúng | 80/10/10 (hoặc official split IMDb), không thiếu/trùng split | Đếm `df.split.value_counts()` |
| D4 | Không leakage | `set(train.id) ∩ set(val.id) == ∅` và tương tự test | Script kiểm intersection id |
| D5 | Không duplicate chéo split | Text giống hệ thống không xuất hiện ở 2 split | Hash text + so sánh giữa splits |
| D6 | Test set cô lập | Không file nào trong `src/training/` đọc split `test` | `grep -r "test" src/training/` — chỉ được dùng ở evaluation |
| D7 | Preprocessing không fit trên test | Tokenizer/scaler chỉ fit trên train | Đọc code, kiểm tra dòng `.fit(` |
| D8 | Metadata processed | `data/processed/metadata.json` ghi seed, row counts, hash split | Mở file kiểm tra |

## 2. Audit `src/models/` + `src/training/` (Member 3) — Reproducibility

| # | Mục kiểm tra | Tiêu chí PASS | Lệnh/Cách kiểm |
|---|---|---|---|
| T1 | Config-driven | Mọi hyperparameter (lr, epochs, batch, seed...) đọc từ `configs/*.yaml`, không hard-code trong `.py` | `grep -rn "2e-5\|batch_size=16" src/` phải trả về 0 kết quả logic |
| T2 | Seed đủ 3 tầng | `random.seed`, `np.random.seed`, `torch.manual_seed` (+ cuda) đều được set từ config | Đọc `src/training/config.py` hoặc `train.py` |
| T3 | Determinism check | Chạy lại cùng config 2 lần, validation loss trùng khớp (hoặc lệch không đổi trong tolerance đã ghi) | Chạy `python -m src.training.train --config configs/bert_finetune.yaml` 2 lần |
| T4 | Checkpoint metadata | Checkpoint lưu đủ 6 trường: weights, tokenizer, config, training state (optimizer/scheduler), epoch, best validation metric | `torch.load(...)` kiểm keys |
| T5 | Model-agnostic loop | Training loop không branch theo tên model cụ thể; factory nhận `model_name, strategy, dropout` | Đọc `src/training/train.py` + `src/models/factory.py` |
| T6 | Frozen strategy đúng | `frozen`: `requires_grad=False` cho toàn encoder, chỉ head train; `finetune`: tất cả train | In `sum(p.numel() for p in encoder.parameters() if p.requires_grad)` = 0 khi frozen |
| T7 | Output range | Prediction nằm trong [0,1] (sigmoid head) | Forward 1 batch, kiểm min/max |
| T8 | Không fake artifact | Mọi file trong `artifacts/` sinh từ run thật, có timestamp + config copy kèm | So `config.yaml` trong artifact với config gốc |

## 3. Audit `src/evaluation/` (Member 4 tự viết — tự kiểm)

| # | Mục kiểm tra | Tiêu chí PASS |
|---|---|---|
| E1 | Metric functions | MAE/MSE/RMSE/R² khớp `sklearn.metrics` trên dữ liệu đối chiếu |
| E2 | Resource metrics | Train time, inference latency, peak memory, param count được ghi vào `metrics.json` |
| E3 | Đủ 6 plots | loss, actual-vs-predicted, residual, error buckets, model comparison, trade-off — file tồn tại trong `artifacts/<exp>/plots/` |
| E4 | Artifact bundle | Mỗi experiment có đủ 5 file: `config.yaml`, `metrics.json`, `training_history.csv`, `predictions.csv`, `model_metadata.json` |
| E5 | Tên experiment | Đúng convention `{model}_{strategy}_{dataset}_seed{seed}` theo `naming_convention.md` |

## 4. Audit Full pipeline (Member 5) — phục vụ tiêu chí 5.2

| # | Mục kiểm tra | Tiêu chí PASS |
|---|---|---|
| P1 | API contract | `POST /predict` nhận `{"text": "..."}`, trả `{"score": 0.x}`, có validation input rỗng |
| P2 | Checkpoint reload | Model load 1 lần lúc startup, restart không đổi kết quả trên cùng input |
| P3 | E2E smoke | 1 review mẫu đi từ UI/API → score trong 1 lần chạy, ghi lại latency thực tế |
| P4 | Trùng khớp score | Score qua API khớp score chạy trực tiếp `src/evaluation/` (cùng checkpoint, cùng input, lệch < 1e-4) |

## 5. Audit quy trình (cross-team)

| # | Mục kiểm tra | Tiêu chí PASS |
|---|---|---|
| W1 | Progress tracking | Mỗi phase xong có tick trong `progress_status.md` kèm bằng chứng kiểm chứng |
| W2 | Docs phản ánh code | `docs/COMMANDS.md` chạy được 100% lệnh với code hiện tại |
| W3 | Không file phiên bản | Không tồn tại `*_v2.py`, `*final_new.py` theo AGENTS.md "Never" |
| W4 | Không path tuyệt đối | `grep -rn "/home/" src/` trả về 0 kết quả |

## Audit log

Mỗi lần audit ghi theo mẫu trong `codebase_audit.md`:

```text
Date:
Auditor: Member 4
Commit:
Critical issues:
Major issues:
Minor issues:
Refactoring:
Tests:
```

### Nhật ký

_(chưa có audit run nào — ghi vào đây khi Member 2/3 có code)_
