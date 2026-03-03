/**
 * Bingo Card Generator
 *
 * Generates balanced bingo cards using Monte Carlo simulation.
 * Ported from Python utils/create_cards.py to JavaScript.
 * Runs entirely in the browser without server.
 */

/**
 * Generate candidate events from the feature counts matrix.
 *
 * @param {object[]} images - Array of images with feature counts from dataset.json
 * @param {string[]} allFeatures - List of all feature names
 * @returns {object} { events: [...], truthMatrix: Float32Array }
 */
function generateEvents(images, allFeatures) {
  const events = [];
  const truthMatrix = [];

  for (const feature of allFeatures) {
    // Extract counts for this feature across all images
    const counts = images.map(img => img.features[feature] || 0);
    const countsArray = new Uint32Array(counts);
    const maxVal = Math.max(...counts);

    // 1. "Contains {feature}" event
    const containsMask = countsArray.map(c => c > 0 ? 1 : 0);
    const containsProb = containsMask.reduce((a, b) => a + b, 0) / containsMask.length;
    if (containsProb >= 0.001 && containsProb <= 0.95) {
      events.push({
        description: `Contains ${feature}`,
        type: 'exists',
        feature: feature,
        probability: containsProb
      });
      truthMatrix.push(containsMask);
    }

    // 2. "More than N {feature}s" events
    if (maxVal > 1) {
      for (let n = 1; n < Math.min(maxVal, 6); n++) {
        const moreMask = countsArray.map(c => c > n ? 1 : 0);
        const moreProb = moreMask.reduce((a, b) => a + b, 0) / moreMask.length;
        if (moreProb >= 0.05 && moreProb <= 0.95) {
          events.push({
            description: `More than ${n} ${feature}s`,
            type: 'threshold',
            feature: feature,
            probability: moreProb
          });
          truthMatrix.push(moreMask);
        }
      }
    }

    // 3. "Exactly N {feature}(s)" events
    if (maxVal >= 1) {
      for (let n = 1; n < Math.min(maxVal + 1, 6); n++) {
        const exactMask = countsArray.map(c => c === n ? 1 : 0);
        const exactProb = exactMask.reduce((a, b) => a + b, 0) / exactMask.length;
        if (exactProb >= 0.05 && exactProb <= 0.95) {
          const singular = n === 1 ? feature : `${feature}s`;
          events.push({
            description: `Exactly ${n} ${singular}`,
            type: 'exact',
            feature: feature,
            probability: exactProb
          });
          truthMatrix.push(exactMask);
        }
      }
    }
  }

  // Convert truthMatrix to 2D array (n_images x n_events)
  const nImages = images.length;
  const nEvents = truthMatrix.length;
  const truthMatrixArray = new Uint8Array(nImages * nEvents);

  for (let i = 0; i < nEvents; i++) {
    for (let j = 0; j < nImages; j++) {
      truthMatrixArray[j * nEvents + i] = truthMatrix[i][j];
    }
  }

  return {
    events: events,
    truthMatrix: { data: truthMatrixArray, nImages: nImages, nEvents: nEvents }
  };
}

/**
 * Calculate average turns to win for multiple cards (vectorized Monte Carlo).
 *
 * @param {number[][]} cardIndicesList - List of event indices for each card
 * @param {object} truthMatrix - { data: Uint8Array, nImages, nEvents }
 * @param {number} nSimulations - Number of Monte Carlo simulations
 * @returns {number[]} Average turns to win for each card
 */
function calculateTurnsToWinVectorized(cardIndicesList, truthMatrix, nSimulations = 1000) {
  const { data: truthData, nImages, nEvents } = truthMatrix;
  const nCards = cardIndicesList.length;

  // Pre-select columns for each card
  const cardMasks = cardIndicesList.map(indices => {
    const mask = new Uint8Array(nImages * indices.length);
    for (let i = 0; i < nImages; i++) {
      for (let j = 0; j < indices.length; j++) {
        mask[i * indices.length + j] = truthData[i * nEvents + indices[j]];
      }
    }
    return mask;
  });

  const turnsNeeded = new Float32Array(nCards);
  const counts = new Float32Array(nCards);

  // Run Monte Carlo simulations
  for (let s = 0; s < nSimulations; s++) {
    // Shuffle image deck
    const deck = Array.from({ length: nImages }, (_, i) => i);
    for (let i = deck.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [deck[i], deck[j]] = [deck[j], deck[i]];
    }

    // For each card, find first turn where all events are covered
    for (let c = 0; c < nCards; c++) {
      const mask = cardMasks[c];
      const cardSize = cardIndicesList[c].length;
      const covered = new Uint8Array(cardSize); // Track which events are covered

      let winTurn = nImages;
      for (let t = 0; t < nImages; t++) {
        const imgIdx = deck[t];
        // Mark events that fire in this image
        for (let e = 0; e < cardSize; e++) {
          if (mask[imgIdx * cardSize + e]) {
            covered[e] = 1;
          }
        }
        // Check if all covered
        if (covered.every(x => x === 1)) {
          winTurn = t + 1;
          break;
        }
      }
      turnsNeeded[c] += winTurn;
    }
  }

  // Average
  return Array.from(turnsNeeded).map(t => t / nSimulations);
}

/**
 * Find a single valid card that meets target difficulty.
 *
 * @param {string[]} uniqueFeatures
 * @param {object} featureGroups - feature -> event indices
 * @param {object} truthMatrix
 * @param {number} cardSize - Number of events per card
 * @param {number} targetDifficulty
 * @param {number} tolerance
 * @param {number} maxAttempts
 * @returns {object|null} { cardIndices, difficulty } or null
 */
function findValidCard(uniqueFeatures, featureGroups, truthMatrix, cardSize, targetDifficulty, tolerance, maxAttempts) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    // Random selection: pick cardSize unique features
    const selectedFeatures = [];
    const availableFeatures = [...uniqueFeatures];
    for (let i = 0; i < cardSize; i++) {
      const randIdx = Math.floor(Math.random() * availableFeatures.length);
      selectedFeatures.push(availableFeatures[randIdx]);
      availableFeatures.splice(randIdx, 1);
    }

    // Pick one event per feature
    const cardIndices = [];
    for (const feat of selectedFeatures) {
      const possibleIndices = featureGroups[feat];
      const randIdx = Math.floor(Math.random() * possibleIndices.length);
      cardIndices.push(possibleIndices[randIdx]);
    }

    // Quick check
    const estDiff = calculateTurnsToWinVectorized([cardIndices], truthMatrix, 50)[0];
    if (Math.abs(estDiff - targetDifficulty) < tolerance * 2) {
      // Precise check
      const preciseDiff = calculateTurnsToWinVectorized([cardIndices], truthMatrix, 5000)[0];
      if (Math.abs(preciseDiff - targetDifficulty) < tolerance) {
        return { cardIndices, difficulty: preciseDiff };
      }
    }
  }
  return null;
}

/**
 * Run tournament simulation to determine win rates.
 *
 * @param {number[][]} cards - List of event indices for each card
 * @param {object} truthMatrix
 * @param {number} nSimulations
 * @returns {Float32Array} Win count for each card
 */
function runTournamentSimulation(cards, truthMatrix, nSimulations) {
  const { data: truthData, nImages, nEvents } = truthMatrix;
  const nCards = cards.length;

  // Pre-select columns for each card
  const cardMasks = cards.map(indices => {
    const mask = new Uint8Array(nImages * indices.length);
    for (let i = 0; i < nImages; i++) {
      for (let j = 0; j < indices.length; j++) {
        mask[i * indices.length + j] = truthData[i * nEvents + indices[j]];
      }
    }
    return mask;
  });

  const winCounts = new Float32Array(nCards);

  // Simulate games
  for (let s = 0; s < nSimulations; s++) {
    // Shuffle deck
    const deck = Array.from({ length: nImages }, (_, i) => i);
    for (let i = deck.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [deck[i], deck[j]] = [deck[j], deck[i]];
    }

    // Find winner(s) for this game
    const turns = new Uint32Array(nCards);
    const didWin = new Uint8Array(nCards);

    for (let c = 0; c < nCards; c++) {
      const mask = cardMasks[c];
      const cardSize = cards[c].length;
      const covered = new Uint8Array(cardSize);

      for (let t = 0; t < nImages; t++) {
        const imgIdx = deck[t];
        for (let e = 0; e < cardSize; e++) {
          if (mask[imgIdx * cardSize + e]) {
            covered[e] = 1;
          }
        }
        if (covered.every(x => x === 1)) {
          turns[c] = t + 1;
          didWin[c] = 1;
          break;
        }
      }
      if (!didWin[c]) {
        turns[c] = nImages + 9999;
      }
    }

    // Award points (split ties)
    const minTurns = Math.min(...turns);
    const winners = turns.map((t, i) => t === minTurns ? 1 : 0);
    const nWinners = winners.reduce((a, b) => a + b, 0);
    for (let c = 0; c < nCards; c++) {
      if (winners[c]) {
        winCounts[c] += 1 / nWinners;
      }
    }
  }

  return winCounts;
}

/**
 * Main function: Generate balanced bingo cards.
 *
 * @param {object[]} images - From dataset.json
 * @param {string[]} allFeatures - All feature names
 * @param {number} numCards - Number of cards to generate
 * @param {number} cardSize - Events per card
 * @param {object} options - { tolerance, targetDifficulty, onProgress }
 * @returns {Promise<array>} Array of card objects
 */
async function generateBalancedCards(images, allFeatures, numCards, cardSize = 10, options = {}) {
  const {
    tolerance = 1,
    targetDifficulty = null,
    onProgress = () => {}
  } = options;

  console.log(`Generating events from ${allFeatures.length} features...`);
  onProgress({ stage: 'generating-events', progress: 0 });

  const { events, truthMatrix } = generateEvents(images, allFeatures);
  console.log(`Generated ${events.length} candidate events`);

  if (events.length < cardSize) {
    throw new Error(`Not enough events (${events.length}) for card size ${cardSize}`);
  }

  // Group events by feature
  const featureGroups = {};
  for (const feat of allFeatures) {
    featureGroups[feat] = events
      .map((e, i) => e.feature === feat ? i : -1)
      .filter(i => i >= 0);
  }

  const uniqueFeatures = Object.keys(featureGroups).filter(f => featureGroups[f].length > 0);

  if (uniqueFeatures.length < cardSize) {
    throw new Error(`Not enough unique features (${uniqueFeatures.length}) for card size ${cardSize}`);
  }

  // Estimate target difficulty if not provided
  let targetDiff = targetDifficulty;
  if (!targetDiff) {
    console.log('Estimating baseline difficulty...');
    onProgress({ stage: 'estimating-difficulty', progress: 0 });

    const sampleCards = [];
    for (let i = 0; i < 50; i++) {
      const feats = [];
      const availFeats = [...uniqueFeatures];
      for (let j = 0; j < cardSize; j++) {
        const idx = Math.floor(Math.random() * availFeats.length);
        feats.push(availFeats[idx]);
        availFeats.splice(idx, 1);
      }
      const indices = feats.map(f => featureGroups[f][Math.floor(Math.random() * featureGroups[f].length)]);
      sampleCards.push(indices);
    }

    const diffs = calculateTurnsToWinVectorized(sampleCards, truthMatrix, 100);
    const validDiffs = diffs.filter(d => d < truthMatrix.nImages);
    if (validDiffs.length === 0) {
      throw new Error('Could not estimate difficulty');
    }
    targetDiff = validDiffs.sort((a, b) => a - b)[Math.floor(validDiffs.length / 2)];
    console.log(`Target difficulty: ${targetDiff.toFixed(2)}`);
  }

  // Generate initial cards in parallel-ish (actually sequential for simplicity)
  console.log(`Generating ${numCards} initial cards...`);
  onProgress({ stage: 'generating-cards', progress: 0, current: 0, total: numCards });

  const finalCards = [];
  const finalStats = [];

  for (let i = 0; i < numCards; i++) {
    const result = findValidCard(uniqueFeatures, featureGroups, truthMatrix, cardSize, targetDiff, tolerance, 5000);
    if (result) {
      finalCards.push(result.cardIndices);
      finalStats.push(result.difficulty);
    }
    onProgress({ stage: 'generating-cards', progress: ((i + 1) / numCards), current: i + 1, total: numCards });
    // Yield to UI
    await new Promise(resolve => setTimeout(resolve, 0));
  }

  if (finalCards.length === 0) {
    throw new Error('Failed to generate any valid cards');
  }

  console.log(`Generated ${finalCards.length} cards, starting balancing loop...`);

  // Balancing loop
  const maxIterations = 50;
  for (let iter = 0; iter < maxIterations; iter++) {
    const currentNum = finalCards.length;
    if (currentNum === 0) break;

    const targetWinRate = 1.0 / currentNum;
    const winRateTolerance = 0.35 * targetWinRate;

    console.log(`Iteration ${iter + 1}: Running tournament with ${currentNum} cards...`);
    onProgress({ stage: 'balancing', iteration: iter + 1, maxIterations });

    // Run tournament
    const totalSims = 50000; // Reduced from 100000 for browser performance
    const winCounts = runTournamentSimulation(finalCards, truthMatrix, totalSims);
    const winRates = Array.from(winCounts).map(w => w / totalSims);

    // Find bad cards
    const badIndices = winRates
      .map((rate, i) => ({ i, deviation: Math.abs(rate - targetWinRate) }))
      .filter(x => x.deviation > winRateTolerance)
      .map(x => x.i);

    if (badIndices.length === 0) {
      console.log('Converged! All cards balanced.');
      break;
    }

    console.log(`Found ${badIndices.length} unbalanced cards, regenerating...`);

    // Keep good cards
    const goodIndices = Array.from({ length: currentNum }, (_, i) => i)
      .filter(i => !badIndices.includes(i));
    const newCards = goodIndices.map(i => finalCards[i]);
    const newStats = goodIndices.map(i => finalStats[i]);

    // Regenerate bad cards
    const needed = numCards - newCards.length;
    for (let i = 0; i < needed; i++) {
      const result = findValidCard(uniqueFeatures, featureGroups, truthMatrix, cardSize, targetDiff, tolerance, 5000);
      if (result) {
        newCards.push(result.cardIndices);
        newStats.push(result.difficulty);
      }
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    Object.assign(finalCards, newCards);
    finalStats.length = 0;
    finalStats.push(...newStats);
  }

  // Convert to card format
  const cardsData = finalCards.map((indices, i) => ({
    card_id: i + 1,
    events: indices.map(idx => events[idx].description),
    avg_turns_to_win_isolation: finalStats[i]
  }));

  console.log(`Generated ${cardsData.length} balanced cards`);
  onProgress({ stage: 'complete', progress: 1 });

  return cardsData;
}
