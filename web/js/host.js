/**
 * Host view controller for Satellite Bingo.
 *
 * Handles PeerJS initialization, lobby management, dynamic card generation,
 * game state, bingo verification, progress tracking, and end-game chart rendering.
 */

let gameState = null;
let dataset = null;
let cards = null;
let peerManager = null;
let myPeerId = null;

const playerList = new Map();
let playerCardMap = new Map();
const playerProgressHistory = new Map();
let verifiedWinnerCount = 0;
let verifiedWinners = [];
let gameStarted = false;
let shortGameCode = null;

/**
 * Generates a random 6-character alphanumeric code (A–Z, 0–9).
 *
 * @returns {string}
 */
function generateGameCode() {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let code = '';
  for (let i = 0; i < 6; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return code;
}

/**
 * Initializes the host view: loads the dataset and opens the PeerJS connection.
 *
 * @returns {Promise<void>}
 */
async function initGame() {
  try {
    dataset = await loadDataset();

    try {
      shortGameCode = generateGameCode();

      peerManager = new PeerManager();

      if (!peerManager.peer) {
        const customPeerId = shortGameCode;
        peerManager.peer = new Peer(customPeerId, {
          debug: 1,
          config: {
            iceServers: [
              { urls: 'stun:stun.l.google.com:19302' },
              { urls: 'stun:stun1.l.google.com:19302' }
            ]
          }
        });
      }

      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('PeerJS initialization timeout')), 10000);
        peerManager.peer.on('open', () => {
          clearTimeout(timeout);
          resolve();
        });
        peerManager.peer.on('error', (err) => {
          clearTimeout(timeout);
          reject(err);
        });
      });

      myPeerId = peerManager.peer.id;

      document.getElementById('gameCodeDisplay').textContent = shortGameCode;

      peerManager.peer.on('connection', (conn) => {
        handlePlayerConnection(conn);
      });

      peerManager.isHost = true;
      peerManager.playerConnections = new Map();

      document.getElementById('lobbySection').style.display = 'block';
    } catch (error) {
      console.error('Failed to initialize PeerJS:', error);
      alert('Failed to initialize P2P connection: ' + error.message);
    }
  } catch (error) {
    console.error('Failed to initialize game:', error);
    alert('Failed to load game data. Please refresh the page.');
  }
}

/**
 * Copies the game code to the clipboard and briefly updates the button label.
 */
function copyGameCode() {
  const code = document.getElementById('gameCodeDisplay').textContent;
  navigator.clipboard.writeText(code).then(() => {
    const btn = document.getElementById('copyGameCodeBtn');
    btn.textContent = '✓ Copied!';
    setTimeout(() => { btn.textContent = '📋 Copy'; }, 2000);
  });
}

/**
 * Attaches event listeners to an incoming player connection.
 *
 * @param {DataConnection} conn - The incoming PeerJS connection.
 */
function handlePlayerConnection(conn) {
  conn.on('open', () => {
    peerManager.playerConnections.set(conn.peer, conn);
  });

  conn.on('data', (data) => {
    handlePlayerMessage(data, conn.peer);
  });

  conn.on('close', () => {
    playerList.delete(conn.peer);
    peerManager.playerConnections.delete(conn.peer);
    updateLobbyDisplay();
  });

  conn.on('error', () => {
    playerList.delete(conn.peer);
    peerManager.playerConnections.delete(conn.peer);
    updateLobbyDisplay();
  });
}

/**
 * Handles a message received from a player.
 *
 * @param {object} data - The message payload.
 * @param {string} peerId - The sender's peer ID.
 */
function handlePlayerMessage(data, peerId) {
  if (data.type === 'player-hello') {
    const playerName = data.name || 'Player';
    playerList.set(peerId, {
      name: playerName,
      ready: true,
      conn: peerManager.playerConnections.get(peerId)
    });
    updateLobbyDisplay();
    broadcastLobbyUpdate();

  } else if (data.type === 'claim-bingo') {
    if (!gameStarted) return;

    const cardId = playerCardMap.get(peerId);
    if (!cardId) return;

    const verified = gameState.checkBingo(cardId);
    if (verified) {
      verifiedWinnerCount++;
      const playerName = playerList.get(peerId)?.name || 'Player';
      verifiedWinners.push({ peerId, name: playerName, turn: gameState.currentTurn });

      peerManager.broadcastMessage({
        type: 'bingo-verified',
        peerId,
        name: playerName,
        winnerNum: verifiedWinnerCount
      });

      updateWinnerDisplay();

      if (verifiedWinnerCount >= 3) {
        triggerGameOver();
      }
    } else {
      peerManager.sendToPlayer(peerId, {
        type: 'bingo-rejected',
        peerId
      });
    }
  }
}

/**
 * Re-renders the lobby player list and updates the start button state.
 */
function updateLobbyDisplay() {
  const count = playerList.size;
  document.getElementById('lobbyPlayerCount').textContent = count;

  let html = '';
  for (const [, player] of playerList) {
    const readyIcon = player.ready ? '✓' : '⏳';
    html += `<div style="padding: 8px 0; border-bottom: 1px solid #e5e7eb;">
      ${readyIcon} <strong>${player.name}</strong>
    </div>`;
  }

  if (count === 0) {
    html = '<p style="color: #6b7280; margin: 0;">Waiting for players...</p>';
  }

  document.getElementById('lobbyPlayerList').innerHTML = html;

  const readyPlayers = Array.from(playerList.values()).filter(p => p.ready).length;
  const startBtn = document.getElementById('startGameBtn');
  if (readyPlayers >= 2) {
    startBtn.disabled = false;
    startBtn.textContent = `🚀 Start Game (${readyPlayers} ready)`;
  } else {
    startBtn.disabled = true;
    startBtn.textContent = `🚀 Start Game (Need ≥2 ready players)`;
  }
}

/**
 * Broadcasts the current lobby player list to all connected players.
 */
function broadcastLobbyUpdate() {
  const players = Array.from(playerList.values()).map((p) => ({
    name: p.name,
    ready: p.ready
  }));

  peerManager.broadcastMessage({
    type: 'lobby-update',
    players
  });
}

/**
 * Starts the game: generates cards, assigns them to players, and begins gameplay.
 *
 * @returns {Promise<void>}
 */
async function hostStartGame() {
  const readyPlayers = Array.from(playerList.entries()).filter(([_, p]) => p.ready);
  if (readyPlayers.length < 2) {
    alert('Need at least 2 ready players to start the game');
    return;
  }

  document.getElementById('lobbySection').style.display = 'none';
  document.getElementById('generatingSection').style.display = 'block';

  try {
    const numCards = readyPlayers.length;
    const cardSize = 10;
    const targetDifficulty = null;

    const allFeatures = Object.keys(dataset.images[0].features);

    cards = await generateBalancedCards(
      dataset.images,
      allFeatures,
      numCards,
      cardSize,
      { targetDifficulty }
    );

    gameState = new GameState(dataset, cards);

    for (let i = 0; i < readyPlayers.length; i++) {
      const [peerId] = readyPlayers[i];
      const cardId = cards[i].card_id;
      playerCardMap.set(peerId, cardId);
      playerProgressHistory.set(peerId, []);
    }

    document.getElementById('generatingSection').style.display = 'none';
    document.getElementById('gameSection').style.display = 'block';

    document.getElementById('totalTurns').textContent = dataset.images.length;
    document.getElementById('connectedPlayersCount').textContent = readyPlayers.length;
    updateProgressDisplay();

    const assignments = readyPlayers.map(([peerId, playerInfo], idx) => ({
      peerId,
      name: playerInfo.name,
      cardId: cards[idx].card_id
    }));

    peerManager.broadcastMessage({
      type: 'game-starting',
      assignments
    });

    for (let i = 0; i < readyPlayers.length; i++) {
      const [peerId, playerInfo] = readyPlayers[i];
      const card = cards[i];

      peerManager.sendToPlayer(peerId, {
        type: 'card-assigned',
        card,
        playerName: playerInfo.name,
        cardId: card.card_id
      });
    }

    gameStarted = true;
  } catch (error) {
    console.error('Failed to generate cards:', error);
    alert('Failed to generate cards: ' + error.message);

    document.getElementById('generatingSection').style.display = 'none';
    document.getElementById('lobbySection').style.display = 'block';
  }
}

/**
 * Advances the game by one image, updates all displays, and broadcasts the new state.
 */
function hostAdvanceTurn() {
  if (!gameState) return;
  if (verifiedWinnerCount >= 3) return;

  const image = gameState.advanceTurn();
  if (!image) {
    triggerGameOver();
    return;
  }

  displayImage(image);

  document.getElementById('currentTurn').textContent = gameState.currentTurn + 1;

  showFiredEvents(image);

  for (const [peerId, cardId] of playerCardMap) {
    const markings = gameState.cardMarkings[cardId];
    const count = markings.filter(Boolean).length;

    if (!playerProgressHistory.has(peerId)) {
      playerProgressHistory.set(peerId, []);
    }
    playerProgressHistory.get(peerId).push(count);
  }

  updateProgressDisplay();
  broadcastGameState(image);
}

/**
 * Broadcasts the current game state (image + all player markings) to all players.
 *
 * @param {object} currentImage - The image data for the current turn.
 */
function broadcastGameState(currentImage) {
  if (!peerManager || playerCardMap.size === 0) return;

  const playerMarkings = {};
  for (const [peerId, cardId] of playerCardMap) {
    playerMarkings[peerId] = gameState.cardMarkings[cardId];
  }

  peerManager.broadcastMessage({
    type: 'game-update',
    currentTurn: gameState.currentTurn,
    currentImage: currentImage,
    playerMarkings: playerMarkings
  });
}

/**
 * Displays a satellite image in the game image container.
 *
 * @param {object} imageData - Image data from dataset.json.
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
 * Shows which card events fired for the current image.
 *
 * @param {object} imageData - Image data from dataset.json.
 */
function showFiredEvents(imageData) {
  const eventsSection = document.getElementById('eventsSection');
  const eventsList = document.getElementById('eventsList');

  const firedEvents = new Set();
  for (const card of cards) {
    const fires = evaluateImageForCard(imageData, card);
    fires.forEach((fire, idx) => {
      if (fire) {
        firedEvents.add(card.events[idx]);
      }
    });
  }

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
 * Re-renders the per-player progress bars in the game section.
 */
function updateProgressDisplay() {
  const container = document.getElementById('cardProgressContainer');
  container.innerHTML = '';

  for (const [peerId, cardId] of playerCardMap) {
    const playerName = playerList.get(peerId)?.name || 'Player';
    const markings = gameState.cardMarkings[cardId];
    const marked = markings.filter(Boolean).length;
    const total = markings.length;
    const percentage = (marked / total) * 100;

    const hasBingo = verifiedWinners.some(w => w.peerId === peerId);

    const html = `
      <div class="card-progress-item ${hasBingo ? 'bingo' : ''}">
        <div class="card-id">${playerName}${hasBingo ? ' ✓ BINGO!' : ''}</div>
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
 * Updates the winners section with the current verified winner count.
 */
function updateWinnerDisplay() {
  if (verifiedWinnerCount > 0) {
    document.getElementById('winnersSection').style.display = 'block';
    document.getElementById('winnerMessage').textContent =
      `${verifiedWinnerCount} verified winner${verifiedWinnerCount !== 1 ? 's' : ''}!`;
  }
}

/**
 * Ends the game: disables controls, broadcasts the result, and shows the end-game screen.
 */
function triggerGameOver() {
  document.getElementById('nextImageBtn').disabled = true;

  const progressData = {};
  for (const [peerId, progress] of playerProgressHistory) {
    progressData[peerId] = progress;
  }

  peerManager.broadcastMessage({
    type: 'game-over',
    winners: verifiedWinners,
    progressData: progressData
  });

  document.getElementById('gameSection').style.display = 'none';
  document.getElementById('endGameSection').style.display = 'block';

  let winnersHtml = '<ol>';
  verifiedWinners.forEach((winner) => {
    winnersHtml += `<li>${winner.name} - Turn ${winner.turn}</li>`;
  });
  winnersHtml += '</ol>';
  document.getElementById('endGameWinnersList').innerHTML = winnersHtml;

  const playerNames = {};
  for (const [peerId] of playerCardMap) {
    playerNames[peerId] = playerList.get(peerId)?.name || 'Player';
  }
  renderEndGameChart('endGameChart', progressData, playerNames);
}

/**
 * Reloads the page to start a new game after user confirmation.
 */
function hostResetGame() {
  if (!confirm('Reset the game? All progress will be lost.')) {
    return;
  }

  location.reload();
}

document.addEventListener('DOMContentLoaded', initGame);
