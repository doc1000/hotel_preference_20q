# Hotel Preference 20 Questions Demo

A deliberately small, sequential prototype for choosing between two hotels using:

- normalized pairwise feature differences;
- PCA-derived compound hotel axes;
- a weakly seeded particle prior over user preferences;
- posterior preference PCA;
- expected information gain;
- soft Bayesian-style updates;
- tempering, diffusion, and prior mixing to avoid overconfidence;
- an optional CLI loop that approximates the eventual chat-agent interaction.

Start with [`DISCUSSION.md`](DISCUSSION.md), then read `hotel_preference_demo.py` top-to-bottom.

developed in this conversation:
https://chatgpt.com/share/6ab2ceca-4670-83e8-8c86-31afb0805a7f

## Run

```bash
pip install -r requirements.txt
python hotel_preference_demo.py
```

For a tiny interactive interview:

```bash
python hotel_preference_demo.py --interactive
```
