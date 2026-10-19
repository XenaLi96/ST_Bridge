"""Generate the same explicitly synthetic fixtures as bridge-st make-demo."""

from pathlib import Path
from bridge_st.data import make_demo

if __name__ == "__main__":
    print(make_demo(Path("data/demo")))
