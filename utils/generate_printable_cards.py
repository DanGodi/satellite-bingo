import json
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path
import math
import json
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path
import math

# Defaults (no CLI args)
CARDS_PER_PAGE = 6
PAGE_ROWS = 3  # pages arranged as 3 rows x 2 cols for cards per page
PAGE_COLS = 2
PAGE_SIZE = (11.69, 8.27)  # A4 Landscape in inches
DPI = 300

# Paths (relative to this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = PROJECT_ROOT / "bingo_cards.json"
OUTPUT_DIR = PROJECT_ROOT / "printable_cards"


def create_page(page_num: int, cards_batch: list, card_rows: int = 2, card_cols: int = 5):
    """Renders a single page containing up to CARDS_PER_PAGE cards.
    card_rows x card_cols define the layout inside each card (e.g. 3x5 for 15 items).
    """
    fig, axes = plt.subplots(PAGE_ROWS, PAGE_COLS, figsize=PAGE_SIZE)
    axes = axes.flatten()

    # Hide all axes first
    for ax in axes:
        ax.axis('off')

    for i, card in enumerate(cards_batch):
        ax = axes[i]
        ax.axis('on')

        # Card title
        card_id = card.get('card_id', i + 1)
        ax.text(0.5, 0.95, f"Bingo Card #{card_id}", ha='center', va='top', fontsize=14, weight='bold', transform=ax.transAxes)

        # Prepare table data
        events = card.get('events', [])
        wrapped = ["\n".join(textwrap.wrap(e, width=15)) for e in events]
        total_cells = card_rows * card_cols
        while len(wrapped) < total_cells:
            wrapped.append("")

        table_data = []
        for r in range(card_rows):
            start = r * card_cols
            table_data.append(wrapped[start:start + card_cols])

        table = ax.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            bbox=[0.03, 0.05, 0.94, 0.85]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)

        for key, cell in table.get_celld().items():
            cell.set_linewidth(1)
            cell.set_edgecolor('black')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"bingo_cards_page_{page_num}.jpg"
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not JSON_PATH.exists():
        print(f"Error: {JSON_PATH} not found. Run the card generation notebook first.")
        return

    with open(JSON_PATH, 'r') as f:
        cards_data = json.load(f)

    if not cards_data:
        print('No cards found in JSON.')
        return

    # Determine the internal card grid size, prefer 5 columns
    max_squares = max(len(c.get('events', [])) for c in cards_data)
    card_cols = 5
    card_rows = math.ceil(max_squares / card_cols)

    total_pages = math.ceil(len(cards_data) / CARDS_PER_PAGE)
    print(f"Generating {total_pages} pages into {OUTPUT_DIR} ...")

    for p in range(total_pages):
        start = p * CARDS_PER_PAGE
        end = start + CARDS_PER_PAGE
        batch = cards_data[start:end]
        create_page(p + 1, batch, card_rows=card_rows, card_cols=card_cols)

    print("Done! Check the 'printable_cards' folder.")

if __name__ == "__main__":
    main()