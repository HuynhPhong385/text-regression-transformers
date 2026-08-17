# Commands

## Environment

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Tests

```bash
pytest -q
```

## Data

```bash
python -m src.data.prepare
```

## Baseline

```bash
python -m src.training.run_baseline --config configs/baseline.yaml
```

## Training

```bash
python -m src.training.train --config configs/bert_finetune.yaml
```

## Evaluation

```bash
python -m src.evaluation.evaluate --experiment bert_finetune_imdb_seed42
```

## API

```bash
uvicorn src.api:app --reload
```

## UI

```bash
streamlit run src/ui/app.py
```

Commands are the target interface. Update this document whenever commands change.
