/**
 * Event evaluation logic for Satellite Bingo.
 *
 * Ports the Python parse_event() function to JavaScript.
 * Evaluates event strings like "Contains pool", "More than 2 pools", "Exactly 1 red car"
 * against image feature counts.
 */

/**
 * Evaluate a single event string against image feature counts.
 *
 * Event formats:
 *  - "Contains {feature}"     → feature count > 0
 *  - "More than {n} {features}" → feature count > n
 *  - "Exactly {n} {feature(s)}"  → feature count === n
 *
 * @param {string} eventStr - The event string to evaluate
 * @param {object} featureCounts - Map of feature name -> count (e.g., {"pool": 3, "red car": 1})
 * @returns {boolean} True if the event fires for this image
 */
function parseEvent(eventStr, featureCounts) {
  // Handle "Contains {feature}" events
  if (eventStr.startsWith("Contains ")) {
    const feat = eventStr.slice(9);  // Remove "Contains " prefix
    return (featureCounts[feat] ?? 0) > 0;
  }

  // Handle "More than {n} {features}" events
  const moreMatch = eventStr.match(/^More than (\d+) (.+)$/);
  if (moreMatch) {
    const n = parseInt(moreMatch[1], 10);
    const restStr = moreMatch[2];  // e.g., "pools" or "small boats"

    // Try to find matching feature by stripping trailing "s"
    // This works for regular plurals in the current feature set
    for (const feat of Object.keys(featureCounts)) {
      if (restStr === feat + "s") {
        return featureCounts[feat] > n;
      }
    }
  }

  // Handle "Exactly {n} {feature(s)}" events
  const exactMatch = eventStr.match(/^Exactly (\d+) (.+)$/);
  if (exactMatch) {
    const n = parseInt(exactMatch[1], 10);
    const restStr = exactMatch[2];  // e.g., "pool" or "red cars"

    // Try to find matching feature
    // For n > 1, feature is plural; for n === 1, feature is singular
    const suffix = n > 1 ? "s" : "";
    for (const feat of Object.keys(featureCounts)) {
      if (restStr === feat + suffix) {
        return featureCounts[feat] === n;
      }
    }
  }

  // Event format not recognized
  return false;
}

/**
 * Evaluate all events on a bingo card against a specific image.
 *
 * @param {object} imageData - Image data from dataset.json: {id, filename, features: {...}}
 * @param {object} card - Bingo card from cards.json: {card_id, events: [...], avg_turns_to_win_isolation}
 * @returns {boolean[]} Array of boolean flags, one per event on the card.
 *                      true means the event fires (square should be marked).
 */
function evaluateImageForCard(imageData, card) {
  const results = [];
  for (const eventStr of card.events) {
    const fires = parseEvent(eventStr, imageData.features);
    results.push(fires);
  }
  return results;
}

/**
 * Check if all squares on a card are marked.
 *
 * @param {boolean[]} markings - Array of marked flags (true = marked)
 * @returns {boolean} True if all squares are marked
 */
function checkCardBingo(markings) {
  return markings.every(Boolean);
}
