/**
 * Player view controller for Satellite Bingo.
 *
 * Handles:
 *  - Joining a P2P game with game code and host peer ID
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
let peerManager = null;
let isConnectedToHost = false;

const STORAGE_KEY_PREFIX = 'bingo_player_';

/**
 * Player joins a game using game code and host peer ID.
 */
async function playerJoinGame() {
  const gameCode = document.getElementById('gameCodeInput').value.trim().toUpperCase();
  const hostPeerId = document.getElementById('hostPeerIdInput').value.trim();

  if (!gameCode || !hostPeerId) {
    alert('Please enter both game code and host peer ID');
    return;
  }

  // Show connecting status
  document.getElementById('joinSection').style.display = 'none';
  document.getElementById('connectingSection').style.display = 'block';
  document.getElementById('connectionStatus').textContent = 'Initializing peer connection...';

  try {
    // Initialize peer manager
    document.getElementById('connectionStatus').textContent = 'Initializing your peer ID...';
    peerManager = new PeerManager();
    await peerManager.joinGame(gameCode, hostPeerId);

    document.getElementById('connectionStatus').textContent = 'Connected! Loading cards from host...';

    // Set up message handler to receive cards from host
    peerManager.onMessage((data) => {
      handleHostMessage(data);
    });

    // Request cards from host
    peerManager.broadcastMessage({
      type: 'player-ready',
      gameCode: gameCode
    });

    isConnectedToHost = true;
    console.log('Connected to host, awaiting cards...');
  } catch (error) {
    console.error('Failed to join game:', error);
    console.error('Error details:', error.message, error.type);

    let errorMsg = error.message;
    if (error.message.includes('timeout')) {
      errorMsg = 'Host not responding. Make sure host has shared the CORRECT peer ID and is currently online.';
    } else if (error.message.includes('Signaling server')) {
      errorMsg = 'Cannot reach signaling server. Check your internet connection.';
    }

    document.getElementById('connectionStatus').textContent = `❌ ${errorMsg}`;
    document.getElementById('connectingSection').style.display = 'none';
    document.getElementById('joinSection').style.display = 'block';

    alert(`Connection failed:\n\n${errorMsg}\n\nDebug: ${error.message}`);
  }
}

/**
 * Handle messages from the host.
 */
function handleHostMessage(data) {
  console.log('Received from host:', data.type);

  if (data.type === 'cards-data') {
    // Host sent the cards and dataset
    cards = data.cards;
    dataset = data.dataset;

    // Initialize markings for all cards
    for (const card of cards) {
      cardMarkings[card.card_id] = new Array(card.events.length).fill(false);
    }

    // Hide connecting section, show game section
    document.getElementById('connectingSection').style.display = 'none';
    document.getElementById('gameSection').style.display = 'block';

    // Populate card selector
    const select = document.getElementById('cardSelect');
    for (const card of cards) {
      const option = document.createElement('option');
      option.value = card.card_id;
      option.textContent = `Card ${card.card_id}`;
      select.appendChild(option);
    }

    // Try to restore saved selection and markings
    restorePlayerState();

    console.log('Received cards from host:', cards.length, 'cards');

    // Send confirmation back to host
    peerManager.broadcastMessage({
      type: 'cards-received',
      cardCount: cards.length
    });
  }
}

/**
 * Initialize on page load.
 */
async function initPlayer() {
  // Check if we should join a game or load local cards
  const urlParams = new URLSearchParams(window.location.search);
  const gameCode = urlParams.get('gameCode');
  const hostPeerId = urlParams.get('hostPeerId');

  if (gameCode && hostPeerId) {
    // Auto-join from URL parameters
    document.getElementById('gameCodeInput').value = gameCode;
    document.getElementById('hostPeerIdInput').value = hostPeerId;
    playerJoinGame();
  }

  // Player can also manually enter game code and peer ID
  console.log('Player view ready. Waiting for game code input or URL parameters.');
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
