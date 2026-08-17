# FAQ

## Why regression?

Output is a continuous normalized score rather than only a class.

## Is binary IMDb sentiment enough?

It can technically be trained as regression on 0/1, but rating-based targets are more aligned with continuous regression.

## Why not PhoBERT?

MVP dataset is English. PhoBERT is more suitable for Vietnamese text.

## Can a weak PC run it?

Yes. Start with DistilBERT, shorter sequence length, small batch and CPU-compatible smoke tests.

## Is high R² mandatory?

No. The project is about methodology and comparison. There is no universal R² threshold.

## Can the AI Agent decide everything?

No. Research objectives and methodology remain under human control.
