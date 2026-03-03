/**
 * Setup page controller
 *
 * Handles:
 *  - Player count selection
 *  - Card generation initiation
 *  - Progress tracking
 *  - Saving generated cards to localStorage
 */

let dataset = null;
let allFeatures = null;
let peerManager = null;
let gameInfo = null;

/**
 * Update player count display when slider changes.
 */
function updatePlayerCount(value) {
  document.getElementById('playerCount').textContent = value;
}

/**
 * Log a message to the progress log.
 */
function logProgress(message) {
  const log = document.getElementById('progressLog');
  const timestamp = new Date().toLocaleTimeString();
  log.innerHTML += `[${timestamp}] ${message}\n`;
  log.scrollTop = log.scrollHeight;
}

/**
 * Update progress bar and labels.
 */
function updateProgress(progress, label, stageName) {
  const percent = Math.round(progress * 100);
  document.getElementById('progressBar').style.width = percent + '%';
  document.getElementById('progressPercent').textContent = percent + '%';
  if (label) {
    document.getElementById('progressLabel').textContent = label;
  }
  if (stageName) {
    document.getElementById('progressStage').textContent = stageName;
  }
}

/**
 * Load dataset and features (runs on page load).
 */
async function initSetup() {
  try {
    logProgress('Loading dataset...');
    dataset = await loadDataset();
    allFeatures = Object.keys(dataset.images[0].features).sort();
    logProgress(`✓ Dataset loaded: ${dataset.images.length} images, ${allFeatures.length} features`);
  } catch (error) {
    logProgress(`✗ Error loading dataset: ${error.message}`);
    console.error('Failed to load dataset:', error);
  }
}

/**
 * Start the card generation process.
 */
async function startGeneration() {
  if (!dataset) {
    alert('Dataset not loaded. Please refresh the page.');
    return;
  }

  const numPlayers = parseInt(document.getElementById('playerCountSlider').value, 10);
  const difficulty = document.getElementById('difficultySelect').value;

  // Map difficulty to target
  let targetDifficulty = null;
  switch (difficulty) {
    case 'easy':
      targetDifficulty = 27;
      break;
    case 'medium':
      targetDifficulty = 37;
      break;
    case 'hard':
      targetDifficulty = 47;
      break;
    default:
      targetDifficulty = null; // Auto-calculate
  }

  // Hide form, show progress
  document.getElementById('setupForm').style.display = 'none';
  document.getElementById('progressSection').style.display = 'block';
  document.getElementById('progressLog').innerHTML = '';

  logProgress(`Generating ${numPlayers} balanced bingo cards...`);
  logProgress(`Card size: 10 events`);
  logProgress(`Target difficulty: ${targetDifficulty ? targetDifficulty.toFixed(1) + ' turns' : 'Auto-calculated'}`);
  logProgress('');

  try {
    const cards = await generateBalancedCards(
      dataset.images,
      allFeatures,
      numPlayers,
      10,
      {
        tolerance: 1,
        targetDifficulty: targetDifficulty,
        onProgress: (progress) => {
          const { stage, current, total, iteration, maxIterations } = progress;

          switch (stage) {
            case 'generating-events':
              updateProgress(0, 'Generating candidate events...', 'Phase 1: Event Generation');
              break;
            case 'estimating-difficulty':
              updateProgress(0.1, 'Estimating baseline difficulty...', 'Phase 2: Difficulty Estimation');
              break;
            case 'generating-cards':
              const cardProgress = current / total;
              updateProgress(0.2 + cardProgress * 0.3, `Generating cards: ${current}/${total}`, 'Phase 3: Initial Generation');
              if (current === total) {
                logProgress(`✓ Generated ${current} initial cards`);
              }
              break;
            case 'balancing':
              const balanceProgress = iteration / maxIterations;
              updateProgress(0.5 + balanceProgress * 0.4, `Balancing: iteration ${iteration}/${maxIterations}`, 'Phase 4: Tournament Balancing');
              if (iteration === 1) {
                logProgress('Running tournament simulations to balance win rates...');
              }
              logProgress(`  Iteration ${iteration}/${maxIterations}`);
              break;
            case 'complete':
              updateProgress(1, 'Complete!', 'Generation Complete');
              logProgress('✓ Card generation complete!');
              break;
          }
        }
      }
    );

    // Save to localStorage
    localStorage.setItem('bingo_generated_cards', JSON.stringify(cards));
    logProgress('✓ Cards saved to browser storage');

    // Show success screen
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('successSection').style.display = 'block';
    document.getElementById('successCardCount').textContent = cards.length;

    // Also update game.js to use these cards
    window.generatedCards = cards;

    // Generate game code and peer ID locally (no network call yet)
    // Host.html will make the actual PeerJS connection
    // Generate a short game code for display/identification only.
    // The actual peer ID is assigned by PeerJS when host.html loads.
    const helper = new PeerManager();
    const gameCode = helper.generateGameCode();
    gameInfo = { gameCode };
    document.getElementById('gameCodeDisplay').textContent = gameCode;
    logProgress('✓ Game session created: ' + gameCode);

  } catch (error) {
    logProgress(`✗ Error: ${error.message}`);
    console.error('Generation failed:', error);
    alert(`Generation failed: ${error.message}`);
    // Show form again
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('setupForm').style.display = 'block';
  }
}

/**
 * Start the host game and navigate to host.html
 */
function startHostGame() {
  if (!gameInfo || !gameInfo.gameCode) {
    alert('Game not initialized. Please refresh and try again.');
    return;
  }

  // Store game info in sessionStorage for host.html to access
  sessionStorage.setItem('hostGameInfo', JSON.stringify(gameInfo));
  sessionStorage.setItem('hostGameCards', JSON.stringify(window.generatedCards));

  // Navigate to host view
  window.location.href = 'host.html';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initSetup);
