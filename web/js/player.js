/**
 * Player view controller for Satellite Bingo - Lobby Edition
 *
 * Handles:
 *  - Joining a game with name entry
 *  - Waiting lobby display
 *  - Loading assigned card from host
 *  - Game play with auto-marked + manual squares
 *  - Claim BINGO verification flow
 *  - Game over screen with chart
 *  - Progress tracking
 */

let dataset = null;
let cards = null;
let selectedCardId = null;
let cardMarkings = {};  // cardId -> boolean[]
let peerManager = null;
let isConnectedToHost = false;
let myPeerId = null;
let myName = null;
let assignedCard = null;
let gameStarted = false;
let playerProgressHistory = [];  // [0, 1, 2, 2, ...] = squares marked per turn
let gameOverData = null; // {winners, progressData}

const STORAGE_KEY_PREFIX = 'bingo_player_';

/**
 * Player joins a game using game code and their name.
 */
async function playerJoinGame() {
  const gameCode = document.getElementById('gameCodeInput').value.trim();
  const playerName = document.getElementById('playerNameInput').value.trim();

  if (!gameCode) {
    alert('Please enter the Game Code from the host');
    return;
  }

  if (!playerName) {
    alert('Please enter your name');
    return;
  }

  if (playerName.length > 20) {
    alert('Name must be 20 characters or less');
    return;
  }

  // Store name
  myName = playerName;

  // Show waiting section
  document.getElementById('joinSection').style.display = 'none';
  document.getElementById('waitingSection').style.display = 'block';
  document.getElementById('waitingPlayerList').innerHTML = '<p style="color: #6b7280;">Connecting...</p>';

  try {
    peerManager = new PeerManager();
    await peerManager.joinGame(gameCode, gameCode); // game code IS the peer ID

    myPeerId = peerManager.peer.id;
    console.log('Connected to host. My peer ID:', myPeerId);

    // Set up message handler
    peerManager.onMessage((data) => {
      handleHostMessage(data);
    });

    // Send hello with name
    peerManager.broadcastMessage({
      type: 'player-hello',
      name: myName
    });

    isConnectedToHost = true;
    console.log('Connected to host. Waiting for game to start...');
  } catch (error) {
    console.error('Failed to join game:', error);
    console.error('Error details:', error.message);

    let errorMsg = error.message;
    if (error.message.includes('timeout')) {
      errorMsg = 'Host not responding. Make sure you entered the CORRECT game code.';
    } else if (error.message.includes('Signaling server')) {
      errorMsg = 'Cannot reach signaling server. Check your internet connection.';
    }

    document.getElementById('waitingSection').style.display = 'none';
    document.getElementById('joinSection').style.display = 'block';

    alert(`Connection failed:\n\n${errorMsg}`);
  }
}

/**
 * Handle messages from the host.
 */
function handleHostMessage(data) {
  console.log('Received from host:', data.type);

  if (data.type === 'lobby-update') {
    // Update player list in waiting section
    updateWaitingLobbyDisplay(data.players);

  } else if (data.type === 'card-assigned') {
    // Host has assigned us a card
    assignedCard = data.card;
    selectedCardId = data.card.card_id;

    // Initialize markings for this card
    cardMarkings[selectedCardId] = new Array(data.card.events.length).fill(false);

    // Load dataset if needed
    if (!dataset) {
      loadDataset().then(() => {
        transitionToGameSection();
      }).catch(err => {
        console.error('Failed to load dataset:', err);
      });
    } else {
      transitionToGameSection();
    }

  } else if (data.type === 'game-update') {
    // Host sent game state update
    if (data.playerMarkings && data.playerMarkings[myPeerId]) {
      const hostMarkings = data.playerMarkings[myPeerId];
      if (cardMarkings[selectedCardId]) {
        // OR the host's markings into ours (keep all previous marks)
        for (let i = 0; i < hostMarkings.length; i++) {
          if (hostMarkings[i]) {
            cardMarkings[selectedCardId][i] = true;
          }
        }

        // Track progress
        const count = cardMarkings[selectedCardId].filter(Boolean).length;
        playerProgressHistory.push(count);

        // Update display
        displayCard(selectedCardId);
        savePlayerState();
      }
    }

  } else if (data.type === 'bingo-verified') {
    // Someone claimed BINGO and it was verified
    console.log('BINGO verified for:', data.name);
    if (data.peerId === myPeerId) {
      showClaimStatus(`✓ BINGO Verified! You are winner #${data.winnerNum}!`, true);
    }

  } else if (data.type === 'bingo-rejected') {
    // Our BINGO claim was rejected
    console.log('BINGO rejected');
    showClaimStatus('✗ BINGO not yet! Keep playing.', false);

  } else if (data.type === 'game-over') {
    // Game has ended
    gameOverData = data;
    transitionToEndGameSection(data);
  }
}

/**
 * Update the waiting lobby display with current players.
 */
function updateWaitingLobbyDisplay(players) {
  let html = '';
  if (players && players.length > 0) {
    for (const player of players) {
      const readyIcon = player.ready ? '✓' : '⏳';
      html += `<div style="padding: 8px 0; border-bottom: 1px solid #e5e7eb;">
        ${readyIcon} <strong>${player.name}</strong>
      </div>`;
    }
  } else {
    html = '<p style="color: #6b7280; margin: 0;">Waiting for other players...</p>';
  }

  document.getElementById('waitingPlayerList').innerHTML = html;
}

/**
 * Transition from waiting to game section.
 */
function transitionToGameSection() {
  document.getElementById('waitingSection').style.display = 'none';
  document.getElementById('gameSection').style.display = 'block';
  gameStarted = true;
  displayCard(selectedCardId);
}

/**
 * Display a specific bingo card.
 */
function displayCard(cardId) {
  if (!cardMarkings[cardId]) {
    console.error('No markings for card', cardId);
    return;
  }

  const card = assignedCard || { events: [] };

  // Update title
  document.getElementById('cardTitle').textContent = `Your Card`;

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

  // Check if all marked (BINGO ready)
  const allMarked = markings.every(Boolean);
  document.getElementById('bingoLabel').style.display = allMarked ? 'block' : 'none';
  document.getElementById('claimBingoBtn').disabled = !allMarked;

  if (allMarked) {
    document.getElementById('bingoCard').classList.add('completed');
  } else {
    document.getElementById('bingoCard').classList.remove('completed');
  }
}

/**
 * Toggle a square's marked state (manual marking).
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
  if (!selectedCardId || !gameStarted) return;

  const markings = cardMarkings[selectedCardId];
  if (!markings.every(Boolean)) {
    alert('Not all squares are marked yet!');
    return;
  }

  // Send BINGO claim to host
  console.log('Claiming BINGO...');
  peerManager.broadcastMessage({
    type: 'claim-bingo',
    cardId: selectedCardId
  });

  showClaimStatus('⏳ Waiting for host verification...', null);
  document.getElementById('claimBingoBtn').disabled = true;
}

/**
 * Show claim status message.
 */
function showClaimStatus(message, isSuccess) {
  const div = document.getElementById('claimStatusDiv');
  const text = document.getElementById('claimStatusText');

  if (isSuccess === true) {
    div.style.background = '#dcfce7';
    div.style.borderTop = '3px solid #16a34a';
    text.style.color = '#15803d';
  } else if (isSuccess === false) {
    div.style.background = '#fee2e2';
    div.style.borderTop = '3px solid #dc2626';
    text.style.color = '#991b1b';
  } else {
    div.style.background = '#e0f2fe';
    div.style.borderTop = '3px solid #0284c7';
    text.style.color = '#0c4a6e';
  }

  text.textContent = message;
  div.style.display = 'block';
}

/**
 * Transition to end game section.
 */
function transitionToEndGameSection(data) {
  document.getElementById('gameSection').style.display = 'none';
  document.getElementById('endGameSection').style.display = 'block';

  // Determine player's rank
  const myRank = data.winners.findIndex(w => w.peerId === myPeerId) + 1;
  const rankText = myRank > 0
    ? `🏆 You placed #${myRank}!`
    : '🎮 You finished the game!';

  document.getElementById('playerRankText').textContent = rankText;

  // Display winners
  let winnersHtml = '<ol>';
  data.winners.forEach((winner, idx) => {
    winnersHtml += `<li>${winner.name} - Turn ${winner.turn}</li>`;
  });
  winnersHtml += '</ol>';
  document.getElementById('endGameWinnersList').innerHTML = winnersHtml;

  // Render chart
  renderEndGameChart(data.progressData);
}

/**
 * Render the end-game chart showing all players' progress.
 */
function renderEndGameChart(progressData) {
  const ctx = document.getElementById('endGameChart');
  if (!ctx) {
    console.error('Chart canvas not found');
    return;
  }

  // Build list of all players for legend
  const allPlayerIds = Object.keys(progressData);
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];

  // Find max turns
  let maxTurns = 0;
  for (const progress of Object.values(progressData)) {
    if (progress.length > maxTurns) maxTurns = progress.length;
  }

  // Build datasets
  const datasets = [];
  allPlayerIds.forEach((peerId, index) => {
    const progress = progressData[peerId];
    const color = colors[index % colors.length];

    // Try to get player name from gameOverData if available
    let playerName = 'Player';
    if (gameOverData && gameOverData.winners) {
      const winner = gameOverData.winners.find(w => w.peerId === peerId);
      if (winner) playerName = winner.name;
    }
    if (peerId === myPeerId) playerName = `${playerName} (You)`;

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
  });

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

/**
 * Initialize on page load.
 */
async function initPlayer() {
  console.log('Player view ready. Waiting for game code input.');
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initPlayer);
