/**
 * Host view controller for Satellite Bingo - Lobby Edition
 *
 * Handles:
 *  - PeerJS initialization and game code display
 *  - Lobby management (player joining with names, ready status)
 *  - Dynamic card generation (1 card per ready player)
 *  - Game state management
 *  - Bingo verification and win detection
 *  - Game over chart rendering
 *  - Progress tracking per player
 */

let gameState = null;
let dataset = null;
let cards = null;
let peerManager = null;
let myPeerId = null;
let difficulty = 'auto';

// Lobby state
const playerList = new Map();  // peerId → { name, ready, conn }
let playerCardMap = new Map(); // peerId → cardId
const playerProgressHistory = new Map(); // peerId → number[] (squares marked per turn)
let verifiedWinnerCount = 0;
let verifiedWinners = []; // [{peerId, name, turn}]
let gameStarted = false;

/**
 * Initialize the game on page load.
 */
async function initGame() {
  try {
    console.log('Host: initGame() starting...');

    // Load data
    console.log('Loading dataset...');
    dataset = await loadDataset();
    console.log('Dataset loaded');

    // Initialize PeerJS immediately
    try {
      console.log('Initializing PeerJS...');
      peerManager = new PeerManager();
      await peerManager.initializePeer();

      myPeerId = peerManager.peer.id;
      console.log('✓ PeerJS initialized with ID:', myPeerId);

      // Display game code
      document.getElementById('gameCodeDisplay').textContent = myPeerId;

      // Set up incoming connections
      peerManager.peer.on('connection', (conn) => {
        console.log('Incoming connection from:', conn.peer);
        handlePlayerConnection(conn);
      });

      peerManager.isHost = true;
      peerManager.playerConnections = new Map();

      // Show lobby section
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
 * Copy game code to clipboard.
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
 * Handle incoming player connection.
 */
function handlePlayerConnection(conn) {
  console.log('Player connecting:', conn.peer);

  conn.on('open', () => {
    console.log('Connection open:', conn.peer);
    peerManager.playerConnections.set(conn.peer, conn);
  });

  conn.on('data', (data) => {
    handlePlayerMessage(data, conn.peer);
  });

  conn.on('close', () => {
    console.log('Player disconnected:', conn.peer);
    playerList.delete(conn.peer);
    peerManager.playerConnections.delete(conn.peer);
    updateLobbyDisplay();
  });

  conn.on('error', (err) => {
    console.error('Connection error with player:', err);
    playerList.delete(conn.peer);
    peerManager.playerConnections.delete(conn.peer);
    updateLobbyDisplay();
  });
}

/**
 * Handle messages from players.
 */
function handlePlayerMessage(data, peerId) {
  console.log('Message from', peerId, ':', data.type);

  if (data.type === 'player-hello') {
    // Player has joined with a name
    const playerName = data.name || 'Player';
    playerList.set(peerId, {
      name: playerName,
      ready: true,
      conn: peerManager.playerConnections.get(peerId)
    });
    updateLobbyDisplay();
    broadcastLobbyUpdate();

  } else if (data.type === 'claim-bingo') {
    // Player is claiming BINGO
    if (!gameStarted) {
      console.warn('Received claim-bingo before game started');
      return;
    }

    const cardId = playerCardMap.get(peerId); // peerId is used here
    if (!cardId) {
      console.warn('Player', peerId, 'not assigned a card');
      return;
    }

    const verified = gameState.checkBingo(cardId);
    if (verified) {
      verifiedWinnerCount++;
      const playerName = playerList.get(peerId)?.name || 'Player';
      verifiedWinners.push({ peerId, name: playerName, turn: gameState.currentTurn });

      console.log('✓ BINGO verified for', playerName, '(', verifiedWinnerCount, 'winners)');

      // Broadcast to all players
      peerManager.broadcastMessage({
        type: 'bingo-verified',
        peerId,
        name: playerName,
        winnerNum: verifiedWinnerCount
      });

      // Update winner display
      updateWinnerDisplay();

      // Check if game should end
      if (verifiedWinnerCount >= 3) {
        triggerGameOver();
      }
    } else {
      console.log('✗ BINGO rejected for', peerId);
      peerManager.sendToPlayer(peerId, {
        type: 'bingo-rejected',
        peerId
      });
    }
  }
}

/**
 * Update the lobby display with current player list.
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

  // Enable start button if enough players are ready
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
 * Broadcast lobby update to all connected players.
 */
function broadcastLobbyUpdate() {
  const players = Array.from(playerList.values()).map((p, idx) => ({
    name: p.name,
    ready: p.ready
  }));

  peerManager.broadcastMessage({
    type: 'lobby-update',
    players
  });
}

/**
 * Host starts the game: generate cards, assign to players, begin gameplay.
 */
async function hostStartGame() {
  console.log('Starting game...');

  difficulty = document.getElementById('difficultySelect').value;

  // Get list of ready players
  const readyPlayers = Array.from(playerList.entries()).filter(([_, p]) => p.ready);
  if (readyPlayers.length < 2) {
    alert('Need at least 2 ready players to start the game');
    return;
  }

  // Disable lobby section, show generating section
  document.getElementById('lobbySection').style.display = 'none';
  document.getElementById('generatingSection').style.display = 'block';

  try {
    // Generate cards
    const numCards = readyPlayers.length;
    const cardSize = 10;

    let targetDifficulty;
    if (difficulty === 'easy') targetDifficulty = 27;
    else if (difficulty === 'medium') targetDifficulty = 37;
    else if (difficulty === 'hard') targetDifficulty = 47;
    else targetDifficulty = null; // auto

    const allFeatures = Object.keys(dataset.images[0].features);

    console.log(`Generating ${numCards} cards (target difficulty: ${targetDifficulty || 'auto'})...`);

    cards = await generateBalancedCards(
      dataset.images,
      allFeatures,
      numCards,
      cardSize,
      { targetDifficulty }
    );

    console.log('✓ Cards generated:', cards.length);

    // Create game state
    gameState = new GameState(dataset, cards);

    // Assign each player a card
    for (let i = 0; i < readyPlayers.length; i++) {
      const [peerId, playerInfo] = readyPlayers[i];
      const cardId = cards[i].card_id;
      playerCardMap.set(peerId, cardId);
      playerProgressHistory.set(peerId, []);

      console.log(`Assigned card ${cardId} to ${playerInfo.name}`);
    }

    // Hide generating, show game section
    document.getElementById('generatingSection').style.display = 'none';
    document.getElementById('gameSection').style.display = 'block';

    // Update displays
    document.getElementById('totalTurns').textContent = dataset.images.length;
    document.getElementById('connectedPlayersCount').textContent = readyPlayers.length;
    updateProgressDisplay();

    // Broadcast game starting to all players
    const assignments = readyPlayers.map(([peerId, playerInfo], idx) => ({
      peerId,
      name: playerInfo.name,
      cardId: cards[idx].card_id
    }));

    peerManager.broadcastMessage({
      type: 'game-starting',
      assignments
    });

    // Send each player their own card
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
    console.log('✓ Game started!');
  } catch (error) {
    console.error('Failed to generate cards:', error);
    alert('Failed to generate cards: ' + error.message);

    // Go back to lobby
    document.getElementById('generatingSection').style.display = 'none';
    document.getElementById('lobbySection').style.display = 'block';
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

  if (verifiedWinnerCount >= 3) {
    console.log('Game already over');
    return;
  }

  const image = gameState.advanceTurn();
  if (!image) {
    console.log('Game over - no more images');
    triggerGameOver();
    return;
  }

  // Update image display
  displayImage(image);

  // Update turn counter
  document.getElementById('currentTurn').textContent = gameState.currentTurn + 1;

  // Show fired events
  showFiredEvents(image);

  // Track progress for each player
  for (const [peerId, cardId] of playerCardMap) {
    const markings = gameState.cardMarkings[cardId];
    const count = markings.filter(Boolean).length;

    if (!playerProgressHistory.has(peerId)) {
      playerProgressHistory.set(peerId, []);
    }
    playerProgressHistory.get(peerId).push(count);
  }

  // Update progress display
  updateProgressDisplay();

  // Broadcast to all players
  broadcastGameState(image);
}


/**
 * Broadcast game state to all connected players.
 */
function broadcastGameState(currentImage) {
  if (!peerManager || playerCardMap.size === 0) {
    return;
  }

  // Build player markings: { peerId: markings[] }
  const playerMarkings = {};
  for (const [peerId, cardId] of playerCardMap) {
    playerMarkings[peerId] = gameState.cardMarkings[cardId];
  }

  const message = {
    type: 'game-update',
    currentTurn: gameState.currentTurn,
    currentImage: currentImage,
    playerMarkings: playerMarkings
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
 * Update the progress display for all cards (players).
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

    // Check if this player has verified BINGO
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
 * Update the verified winners display.
 */
function updateWinnerDisplay() {
  if (verifiedWinnerCount > 0) {
    document.getElementById('winnersSection').style.display = 'block';
    document.getElementById('winnerMessage').textContent =
      `${verifiedWinnerCount} verified winner${verifiedWinnerCount !== 1 ? 's' : ''}!`;
  }
}

/**
 * Trigger game over: stop accepting turns, broadcast result, show chart.
 */
function triggerGameOver() {
  console.log('Game Over! Verified winners:', verifiedWinners.length);

  // Disable next button
  document.getElementById('nextImageBtn').disabled = true;

  // Collect progress data
  const progressData = {};
  for (const [peerId, progress] of playerProgressHistory) {
    progressData[peerId] = progress;
  }

  // Broadcast game over to all players
  peerManager.broadcastMessage({
    type: 'game-over',
    winners: verifiedWinners,
    progressData: progressData
  });

  // Show end-game section
  document.getElementById('gameSection').style.display = 'none';
  document.getElementById('endGameSection').style.display = 'block';

  // Populate winners list
  let winnersHtml = '<ol>';
  verifiedWinners.forEach((winner, idx) => {
    winnersHtml += `<li>${winner.name} - Turn ${winner.turn}</li>`;
  });
  winnersHtml += '</ol>';
  document.getElementById('endGameWinnersList').innerHTML = winnersHtml;

  // Render chart
  renderEndGameChart(progressData);
}

/**
 * Render the end-game chart showing player progress.
 */
function renderEndGameChart(progressData) {
  const ctx = document.getElementById('endGameChart');
  if (!ctx) {
    console.error('Chart canvas not found');
    return;
  }

  // Prepare datasets
  const datasets = [];
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

  let maxTurns = 0;
  for (const progress of Object.values(progressData)) {
    if (progress.length > maxTurns) maxTurns = progress.length;
  }

  let colorIdx = 0;
  for (const [peerId, progress] of Object.entries(progressData)) {
    const playerName = playerList.get(peerId)?.name || 'Player';
    const color = colors[colorIdx % colors.length];
    colorIdx++;

    datasets.push({
      label: playerName,
      data: progress,
      borderColor: color,
      backgroundColor: color + '20',
      tension: 0.3,
      fill: false,
      borderWidth: 2,
      pointRadius: 3,
      pointBackgroundColor: color
    });
  }

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array.from({ length: maxTurns }, (_, i) => (i + 1).toString()),
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top'
        },
        title: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 10,
          title: {
            display: true,
            text: 'Squares Completed'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Image Number'
          }
        }
      }
    }
  });
}

/**
 * Reset the game to start over.
 */
function hostResetGame() {
  if (!confirm('Reset the game? All progress will be lost.')) {
    return;
  }

  location.reload();
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initGame);
