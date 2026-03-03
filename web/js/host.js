/**
 * Host view controller for Satellite Bingo.
 *
 * Handles:
 *  - Loading and initializing the game
 *  - Advancing turns and displaying images
 *  - Showing fired events for the current turn
 *  - Displaying progress for all cards
 *  - Detecting and announcing winners
 *  - P2P connections with players
 *  - Broadcasting game state to connected players
 */

let gameState = null;
let dataset = null;
let cards = null;
let peerManager = null;
let hostGameInfo = null;
let connectedPlayers = new Set();

/**
 * Initialize the game on page load.
 */
async function initGame() {
  try {
    console.log('initGame() starting...');

    // Check if game info was passed from setup.html
    const gameInfoStr = sessionStorage.getItem('hostGameInfo');
    const cardsStr = sessionStorage.getItem('hostGameCards');

    if (gameInfoStr && cardsStr) {
      // Use generated cards and game info
      hostGameInfo = JSON.parse(gameInfoStr);
      cards = JSON.parse(cardsStr);
      sessionStorage.removeItem('hostGameInfo');
      sessionStorage.removeItem('hostGameCards');
    }

    // Load data
    console.log('Loading dataset...');
    dataset = await loadDataset();
    console.log('Dataset loaded');

    // If cards weren't passed from setup, load defaults
    if (!cards) {
      console.log('Loading default cards...');
      cards = await getCards();
    }
    console.log('Cards loaded, creating game state...');

    // Create game state
    gameState = new GameState(dataset, cards);
    console.log('GameState created');

    // Initialize P2P if we have game info
    if (hostGameInfo) {
      try {
        peerManager = new PeerManager();
        // Game code already created in setup.js, just set it
        peerManager.gameCode = hostGameInfo.gameCode;
        peerManager.peer = new Peer(hostGameInfo.peerId, {
          debug: 0,
          config: {
            iceServers: [
              { urls: 'stun:stun.l.google.com:19302' },
              { urls: 'stun:stun1.l.google.com:19302' }
            ]
          }
        });

        peerManager.isHost = true;
        peerManager.playerConnections = new Map();

        peerManager.peer.on('connection', (conn) => {
          handlePlayerConnection(conn);
        });

        // Display game code and peer ID
        document.getElementById('gameCodeDisplay').textContent = hostGameInfo.gameCode;
        document.getElementById('hostPeerIdDisplay').textContent = hostGameInfo.peerId;
        document.getElementById('shareInfoSection').style.display = 'block';

        console.log('P2P host initialized. Game code:', hostGameInfo.gameCode);
      } catch (error) {
        console.error('Failed to initialize P2P:', error);
      }
    }

    // Update UI
    console.log('Updating UI...');
    document.getElementById('totalTurns').textContent = dataset.images.length;
    updateProgressDisplay();

    console.log('Game initialized with', dataset.images.length, 'images and', cards.length, 'cards');
  } catch (error) {
    console.error('Failed to initialize game:', error);
    console.error('Error details:', error.message, error.stack);
    alert('Failed to load game data. Please refresh the page.');
  }
}

/**
 * Handle incoming player connection.
 */
function handlePlayerConnection(conn) {
  console.log('Player connecting:', conn.peer);

  conn.on('open', () => {
    connectedPlayers.add(conn.peer);
    console.log('Player connected:', conn.peer);
    updatePlayerCountDisplay();

    // Send cards and dataset to the player
    conn.send({
      type: 'cards-data',
      cards: cards,
      dataset: dataset,
      currentTurn: gameState.currentTurn,
      currentImage: gameState.currentTurn < dataset.images.length ? dataset.images[gameState.currentTurn] : null
    });
  });

  conn.on('data', (data) => {
    handlePlayerMessage(data, conn.peer);
  });

  conn.on('close', () => {
    connectedPlayers.delete(conn.peer);
    console.log('Player disconnected:', conn.peer);
    updatePlayerCountDisplay();
  });

  conn.on('error', (err) => {
    console.error('Connection error with player:', err);
    connectedPlayers.delete(conn.peer);
  });

  // Store connection for broadcasting
  peerManager.playerConnections.set(conn.peer, conn);
}

/**
 * Handle messages from players.
 */
function handlePlayerMessage(data, playerId) {
  console.log('Message from player:', data.type);

  if (data.type === 'player-ready') {
    console.log('Player ready for cards:', playerId);
  } else if (data.type === 'cards-received') {
    console.log('Player received cards:', playerId, 'cards:', data.cardCount);
  }
}

/**
 * Update the player count display.
 */
function updatePlayerCountDisplay() {
  const count = connectedPlayers.size;
  const display = document.getElementById('connectedPlayersCount');
  if (display) {
    display.textContent = count;
  }
}

/**
 * Advance to the next image and update all displays.
 */
function hostAdvanceTurn() {
  if (!gameState) {
    console.error('Game not initialized');
    return;
  }

  const image = gameState.advanceTurn();
  if (!image) {
    console.log('Game over - no more images');
    return;
  }

  // Update image display
  displayImage(image);

  // Update turn counter
  document.getElementById('currentTurn').textContent = gameState.currentTurn + 1;

  // Show fired events
  showFiredEvents(image);

  // Update progress
  updateProgressDisplay();

  // Check for winners
  checkForWinners();

  // Broadcast turn update to all connected players
  broadcastGameState(image);
}

/**
 * Broadcast game state to all connected players.
 */
function broadcastGameState(currentImage) {
  if (!peerManager || connectedPlayers.size === 0) {
    return;
  }

  const message = {
    type: 'game-update',
    currentTurn: gameState.currentTurn,
    currentImage: currentImage,
    cardMarkings: gameState.cardMarkings
  };

  // Send to all connected players
  peerManager.playerConnections.forEach((conn) => {
    if (conn.open) {
      conn.send(message);
    }
  });
}

/**
 * Display a satellite image in the image container.
 *
 * @param {object} imageData
 */
function displayImage(imageData) {
  const imgElement = document.getElementById('gameImage');
  const noImageState = document.getElementById('noImageState');
  const imageIdDisplay = document.getElementById('imageIdDisplay');

  imgElement.src = imageData.filename;
  imgElement.style.display = 'block';
  noImageState.style.display = 'none';
  imageIdDisplay.textContent = `Image: ${imageData.id}`;
  imageIdDisplay.style.display = 'block';
}

/**
 * Show which events fired for the current image.
 *
 * @param {object} imageData
 */
function showFiredEvents(imageData) {
  const eventsSection = document.getElementById('eventsSection');
  const eventsList = document.getElementById('eventsList');

  // Find all events that fire across all cards
  const firedEvents = new Set();
  for (const card of cards) {
    const fires = evaluateImageForCard(imageData, card);
    fires.forEach((fire, idx) => {
      if (fire) {
        firedEvents.add(card.events[idx]);
      }
    });
  }

  // Display them
  if (firedEvents.size > 0) {
    eventsList.innerHTML = Array.from(firedEvents)
      .map(event => `<div class="event-chip">${event}</div>`)
      .join('');
    eventsSection.style.display = 'block';
  } else {
    eventsList.innerHTML = '<p style="color: #6b7280;">No events fired for this image</p>';
    eventsSection.style.display = 'block';
  }
}

/**
 * Update the progress display for all cards.
 */
function updateProgressDisplay() {
  const container = document.getElementById('cardProgressContainer');
  container.innerHTML = '';

  for (const card of cards) {
    const markings = gameState.cardMarkings[card.card_id];
    const marked = markings.filter(Boolean).length;
    const total = markings.length;
    const percentage = (marked / total) * 100;
    const hasBingo = gameState.checkBingo(card.card_id);

    const html = `
      <div class="card-progress-item ${hasBingo ? 'bingo' : ''}">
        <div class="card-id">Card ${card.card_id}${hasBingo ? ' ✓ BINGO!' : ''}</div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${percentage}%"></div>
        </div>
        <div class="progress-text">${marked} / ${total} squares</div>
      </div>
    `;
    container.innerHTML += html;
  }
}

/**
 * Check if any cards have won and announce them.
 */
function checkForWinners() {
  const winners = gameState.getWinners();
  if (winners.length > 0) {
    const winnersSection = document.getElementById('winnersSection');
    const winnerCardId = document.getElementById('winnerCardId');
    winnerCardId.textContent = winners[0].cardId;
    winnersSection.style.display = 'block';

    // Disable next button when someone wins
    document.getElementById('nextImageBtn').disabled = true;
  }
}

/**
 * Reset the game to start over.
 */
function hostResetGame() {
  if (!confirm('Reset the game? All progress will be lost.')) {
    return;
  }

  gameState.clearStorage();
  gameState = new GameState(dataset, cards);

  document.getElementById('currentTurn').textContent = 0;
  document.getElementById('gameImage').style.display = 'none';
  document.getElementById('noImageState').style.display = 'block';
  document.getElementById('imageIdDisplay').style.display = 'none';
  document.getElementById('eventsSection').style.display = 'none';
  document.getElementById('winnersSection').style.display = 'none';
  document.getElementById('nextImageBtn').disabled = false;

  updateProgressDisplay();
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initGame);
