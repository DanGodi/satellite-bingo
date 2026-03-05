# Segmentation Bingo 🛰️

A multiplayer Bingo game played with satellite imagery. An AI model detects objects in each image — ships, cars, buildings, pools — and fires events on your card automatically.

## 🚀 Play Online

**Live game:** https://DanGodi.github.io/segmentation-bingo/

No installation required. One player hosts, shares a game code, and everyone joins from their own device. Works across different networks on desktop, tablet, and mobile.

## 🎮 How to Play

1. **Host** opens `host.html`, gets a game code, and shares it with players.
2. **Players** open `player.html`, enter the game code and their name, and click **Join & Ready**.
3. Once at least two players are ready, the host clicks **Start Game** — each player receives a unique bingo card.
4. The host reveals satellite images one at a time with **Next Image**.
5. When the AI detects objects matching a card event, that square marks automatically. Players can also click squares manually.
6. The first player with all 10 squares marked clicks **CLAIM BINGO!** — the host verifies the win.
7. After 3 verified winners, the game ends and shows a progress chart for all players.

## ⚙️ How It Works

- **No server.** Everything runs in the browser. Multiplayer sync uses [PeerJS](https://peerjs.com/) (WebRTC) for direct peer-to-peer connections.
- **Pre-computed data.** The AI segmentation was run offline and the results are stored in `web/data/dataset.json`. The web app reads that file — it does not run the model live.
- **Fair cards.** Cards are generated at game start using a Monte Carlo algorithm to ensure similar expected win times across all players.
- **Hosted on GitHub Pages.** Push to `main` and the live site updates automatically.

## 🧪 Developer: Build a Custom Dataset

To use your own images, you need to run the Python pipeline offline.

### Requirements

- Python 3.10+, GPU recommended (CUDA or Apple Silicon MPS)
- A [HuggingFace access token](https://huggingface.co/settings/tokens) with access to [facebook/sam3](https://huggingface.co/facebook/sam3)

### Setup

```bash
git clone https://github.com/DanGodi/segmentation-bingo.git
cd segmentation-bingo
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Pipeline

The full pipeline is orchestrated from `run_full_game.ipynb`:

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `utils/process_images.py` | Resize raw images to standard format |
| 2 | `utils/label_images.py` | Interactive widget to tag which features appear in each image |
| 3 | `utils/analyze_segment.py` | Run SAM3 via HuggingFace `transformers`; generate mask GeoTIFFs and `segmentation_stats.csv` |
| 4 | `utils/create_cards.py` | Generate balanced bingo cards from stats |
| 5 | `utils/prepare_web_assets.py` | Package outputs into `web/data/` and `web/images/` |

After step 5, commit the updated `web/data/` and `web/images/` directories and the live site will use your new dataset.

### Model access

SAM3 is a gated model. Before running step 3:

1. Accept the license at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3).
2. Create a read-only access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. In `run_full_game.ipynb`, find **Step 3** and set your token in the login cell.

## 📂 Project Structure

```
segmentation-bingo/
  index.html, host.html, player.html   ← Web app entry points
  web/
    css/style.css                       ← Responsive UI
    js/
      events.js                         ← Event string evaluation
      game.js                           ← GameState class, shared utilities
      host.js, player.js                ← View controllers
      peer-connection.js                ← PeerJS wrapper
      card-generator.js                 ← Monte Carlo card generation
    data/
      dataset.json                      ← Pre-computed image metadata
      cards.json                        ← Default bingo cards
    images/
      image_1.jpg … image_36.jpg        ← Resized satellite images
  utils/                                ← Python pipeline scripts
  notebooks/                            ← Jupyter notebooks
  run_full_game.ipynb                   ← Master pipeline notebook
```
