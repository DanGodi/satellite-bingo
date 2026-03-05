/**
 * Player view controller for Satellite Bingo.
 *
 * Handles joining a game, receiving an assigned card, tracking auto-marked
 * and manual square markings, claiming BINGO, and rendering the end-game chart.
 */

let dataset = null;
let cards = null;
let selectedCardId = null;
let cardMarkings = {};
let peerManager = null;
let isConnectedToHost = false;
let myPeerId = null;
let myName = null;
let assignedCard = null;
let gameStarted = false;
let playerProgressHistory = [];
let gameOverData = null;

const STORAGE_KEY_PREFIX = 'bingo_player_';

/**
 * Joins a game using the host's game code and the player's chosen name.
 *
 * @returns {Promise<void>}
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

  myName = playerName;

  document.getElementById('joinSection').style.display = 'none';
  document.getElementById('waitingSection').style.display = 'block';
  document.getElementById('waitingPlayerList').innerHTML = '<p style="color: #6b7280;">Connecting...</p>';

  try {
    peerManager = new PeerManager();
    await peerManager.joinGame(gameCode, gameCode);

    myPeerId = peerManager.peer.id;

    peerManager.onMessage((data) => {
      handleHostMessage(data);
    });

    peerManager.sendToHost({
      type: 'player-hello',
      name: myName
    });

    isConnectedToHost = true;
  } catch (error) {
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
 * Dispatches incoming messages from the host to the appropriate handler.
 *
 * @param {object} data - The message payload received from the host.
 */
function handleHostMessage(data) {
  if (data.type === 'lobby-update') {
    updateWaitingLobbyDisplay(data.players);

  } else if (data.type === 'card-assigned') {
    assignedCard = data.card;
    selectedCardId = data.card.card_id;

    cardMarkings[selectedCardId] = new Array(data.card.events.length).fill(false);

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
    if (data.playerMarkings && data.playerMarkings[myPeerId]) {
      const hostMarkings = data.playerMarkings[myPeerId];
      if (cardMarkings[selectedCardId]) {
        for (let i = 0; i < hostMarkings.length; i++) {
          if (hostMarkings[i]) {
            cardMarkings[selectedCardId][i] = true;
          }
        }

        const count = cardMarkings[selectedCardId].filter(Boolean).length;
        playerProgressHistory.push(count);

        displayCard(selectedCardId);
        savePlayerState();
      }
    }

  } else if (data.type === 'bingo-verified') {
    if (data.peerId === myPeerId) {
      showClaimStatus(`✓ BINGO Verified! You are winner #${data.winnerNum}!`, true);
    }

  } else if (data.type === 'bingo-rejected') {
    showClaimStatus('✗ BINGO not yet! Keep playing.', false);

  } else if (data.type === 'game-over') {
    gameOverData = data;
    transitionToEndGameSection(data);
  }
}

/**
 * Updates the player list shown in the waiting lobby.
 *
 * @param {Array<{name: string, ready: boolean}>} players - Current lobby players.
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
 * Transitions from the waiting lobby to the active game section.
 */
function transitionToGameSection() {
  document.getElementById('waitingSection').style.display = 'none';
  document.getElementById('gameSection').style.display = 'block';
  gameStarted = true;
  displayCard(selectedCardId);
}

/**
 * Renders the player's assigned bingo card with current markings.
 *
 * @param {number} cardId - The card ID to display.
 */
function displayCard(cardId) {
  if (!cardMarkings[cardId]) {
    console.error('No markings for card', cardId);
    return;
  }

  const card = assignedCard || { events: [] };

  document.getElementById('cardTitle').textContent = `Your Card`;

  const grid = document.getElementById('cardGrid');
  grid.innerHTML = '';

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
 * Toggles a square's marked state (manual player interaction).
 *
 * @param {number} cardId - The card containing the square.
 * @param {number} eventIndex - Index of the square within the card's events array.
 */
function playerToggleSquare(cardId, eventIndex) {
  if (!cardMarkings[cardId]) return;

  cardMarkings[cardId][eventIndex] = !cardMarkings[cardId][eventIndex];
  savePlayerState();
  displayCard(cardId);
}

/**
 * Sends a BINGO claim to the host for verification.
 */
function playerClaimBingo() {
  if (!selectedCardId || !gameStarted) return;

  const markings = cardMarkings[selectedCardId];
  if (!markings.every(Boolean)) {
    alert('Not all squares are marked yet!');
    return;
  }

  peerManager.sendToHost({
    type: 'claim-bingo',
    cardId: selectedCardId
  });

  showClaimStatus('⏳ Waiting for host verification...', null);
  document.getElementById('claimBingoBtn').disabled = true;
}

/**
 * Displays a BINGO claim status message with colour-coded styling.
 *
 * @param {string} message - The status message to display.
 * @param {boolean|null} isSuccess - true for success, false for failure, null for pending.
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
 * Transitions to the end-game section and renders the results chart.
 *
 * @param {object} data - Game-over payload from the host.
 * @param {Array<{peerId: string, name: string, turn: number}>} data.winners - Ordered winners list.
 * @param {Object.<string, number[]>} data.progressData - Per-player progress history.
 */
function transitionToEndGameSection(data) {
  document.getElementById('gameSection').style.display = 'none';
  document.getElementById('endGameSection').style.display = 'block';

  const myRank = data.winners.findIndex(w => w.peerId === myPeerId) + 1;
  const rankText = myRank > 0
    ? `🏆 You placed #${myRank}!`
    : '🎮 You finished the game!';

  document.getElementById('playerRankText').textContent = rankText;

  let winnersHtml = '<ol>';
  data.winners.forEach((winner) => {
    winnersHtml += `<li>${winner.name} - Turn ${winner.turn}</li>`;
  });
  winnersHtml += '</ol>';
  document.getElementById('endGameWinnersList').innerHTML = winnersHtml;

  const playerNames = {};
  for (const w of (data.winners || [])) {
    playerNames[w.peerId] = w.peerId === myPeerId ? `${w.name} (You)` : w.name;
  }
  renderEndGameChart('endGameChart', data.progressData, playerNames);
}

/**
 * Persists the current card markings to localStorage.
 */
function savePlayerState() {
  const state = {
    selectedCardId: selectedCardId,
    cardMarkings: cardMarkings,
  };
  localStorage.setItem(STORAGE_KEY_PREFIX + 'state', JSON.stringify(state));
}

/**
 * Clears all marks on the current card after user confirmation.
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
 * Initializes the player view on page load.
 */
async function initPlayer() {}

document.addEventListener('DOMContentLoaded', initPlayer);
