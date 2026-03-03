/**
 * Player view controller for Satellite Bingo.
 *
 * Handles:
 *  - Loading cards and dataset
 *  - Displaying the player's selected card
 *  - Manual square marking/unmarking
 *  - Detecting and announcing BINGO
 *  - Persisting player card state to localStorage
 */

let dataset = null;
let cards = null;
let selectedCardId = null;
let cardMarkings = {};  // cardId -> boolean[]

const STORAGE_KEY_PREFIX = 'bingo_player_';

/**
 * Initialize on page load.
 */
async function initPlayer() {
  try {
    // Load data
    dataset = await loadDataset();
    cards = await getCards();

    // Populate card selector
    const select = document.getElementById('cardSelect');
    for (const card of cards) {
      const option = document.createElement('option');
      option.value = card.card_id;
      option.textContent = `Card ${card.card_id}`;
      select.appendChild(option);
    }

    // Initialize markings for all cards
    for (const card of cards) {
      cardMarkings[card.card_id] = new Array(card.events.length).fill(false);
    }

    // Try to restore saved selection and markings
    restorePlayerState();

    console.log('Player view initialized with', cards.length, 'cards');
  } catch (error) {
    console.error('Failed to initialize player view:', error);
    alert('Failed to load game data. Please refresh the page.');
  }
}

/**
 * Load previously selected card and markings from localStorage.
 */
function restorePlayerState() {
  const saved = localStorage.getItem(STORAGE_KEY_PREFIX + 'state');
  if (saved) {
    try {
      const state = JSON.parse(saved);
      if (state.selectedCardId && cards.find(c => c.card_id === state.selectedCardId)) {
        selectedCardId = state.selectedCardId;
        cardMarkings = state.cardMarkings || cardMarkings;

        // Update selector
        document.getElementById('cardSelect').value = selectedCardId;
        displayCard(selectedCardId);
      }
    } catch (e) {
      console.error('Failed to restore player state:', e);
    }
  }
}

/**
 * Save player state to localStorage.
 */
function savePlayerState() {
  const state = {
    selectedCardId: selectedCardId,
    cardMarkings: cardMarkings,
  };
  localStorage.setItem(STORAGE_KEY_PREFIX + 'state', JSON.stringify(state));
}

/**
 * Handle card selection from dropdown.
 */
function playerSelectCard() {
  const select = document.getElementById('cardSelect');
  const cardId = parseInt(select.value, 10);

  if (!cardId || !cards.find(c => c.card_id === cardId)) {
    document.getElementById('cardContainer').style.display = 'none';
    return;
  }

  selectedCardId = cardId;
  savePlayerState();
  displayCard(cardId);
}

/**
 * Display a specific bingo card.
 *
 * @param {number} cardId
 */
function displayCard(cardId) {
  const card = cards.find(c => c.card_id === cardId);
  if (!card) return;

  // Update title
  document.getElementById('cardTitle').textContent = `Card ${card.card_id}`;

  // Clear grid
  const grid = document.getElementById('cardGrid');
  grid.innerHTML = '';

  // Add squares
  const markings = cardMarkings[cardId];
  for (let i = 0; i < card.events.length; i++) {
    const square = document.createElement('div');
    square.className = 'bingo-square';
    if (markings[i]) {
      square.classList.add('marked');
    }
    square.textContent = card.events[i];
    square.onclick = () => playerToggleSquare(cardId, i);
    grid.appendChild(square);
  }

  // Check if all marked (bingo)
  const allMarked = markings.every(Boolean);
  document.getElementById('bingoLabel').style.display = allMarked ? 'block' : 'none';
  document.getElementById('claimBingoBtn').disabled = !allMarked;
  if (allMarked) {
    document.getElementById('bingoCard').classList.add('completed');
  } else {
    document.getElementById('bingoCard').classList.remove('completed');
  }

  // Show card container
  document.getElementById('cardContainer').style.display = 'block';
}

/**
 * Toggle a square's marked state (manual marking).
 *
 * @param {number} cardId
 * @param {number} eventIndex
 */
function playerToggleSquare(cardId, eventIndex) {
  if (!cardMarkings[cardId]) return;

  cardMarkings[cardId][eventIndex] = !cardMarkings[cardId][eventIndex];
  savePlayerState();
  displayCard(cardId);
}

/**
 * Handle the CLAIM BINGO button.
 */
function playerClaimBingo() {
  if (!selectedCardId) return;

  const markings = cardMarkings[selectedCardId];
  if (markings.every(Boolean)) {
    alert(`🎉 Card ${selectedCardId} WINS! Notify the host!`);
  }
}

/**
 * Clear all marks on the current card.
 */
function playerClearCard() {
  if (!selectedCardId) return;

  if (!confirm('Clear all marks on this card?')) {
    return;
  }

  cardMarkings[selectedCardId] = new Array(cardMarkings[selectedCardId].length).fill(false);
  savePlayerState();
  displayCard(selectedCardId);
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initPlayer);
