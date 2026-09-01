cd "$(dirname "$0")"
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi
python3 gesture_meme.py
