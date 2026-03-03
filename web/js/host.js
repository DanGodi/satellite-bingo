/**
 * Host view controller for Satellite Bingo.
 *
 * Handles:
 *  - Loading and initializing the game
 *  - Advancing turns and displaying images
 *  - Showing fired events for the current turn
 *  - Displaying progress for all cards
 *  - Detecting and announcing winners
 */

let gameState = null;
let dataset = null;
let cards = null;

/**
 * Initialize the game on page load.
 */
async function initGame() {
  try {
    console.log('initGame() starting...');
    // Load data
    console.log('Loading dataset...');
    dataset = await loadDataset();
    console.log('Dataset loaded, loading cards...');
    cards = await getCards();
    console.log('Cards loaded, creating game state...');

    // Create game state
    gameState = new GameState(dataset, cards);
    console.log('GameState created');

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
