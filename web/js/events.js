/**
 * Event evaluation logic for Satellite Bingo.
 *
 * Ports the Python parse_event() function to JavaScript.
 * Evaluates event strings like "Contains pool", "More than 2 pools", "Exactly 1 red car"
 * against image feature counts from dataset.json.
 */

/**
 * Evaluates a single event string against image feature counts.
 *
 * Supported formats:
 *   - "Contains {feature}"          → feature count > 0
 *   - "More than {n} {features}"    → feature count > n (plural feature name)
 *   - "Exactly {n} {feature(s)}"    → feature count === n
 *
 * @param {string} eventStr - The event string to evaluate.
 * @param {Object.<string, number>} featureCounts - Map of feature name to count.
 * @returns {boolean} True if the event fires for this image.
 */
function parseEvent(eventStr, featureCounts) {
  if (eventStr.startsWith("Contains ")) {
    const feat = eventStr.slice(9);
    return (featureCounts[feat] ?? 0) > 0;
  }

  const moreMatch = eventStr.match(/^More than (\d+) (.+)$/);
  if (moreMatch) {
    const n = parseInt(moreMatch[1], 10);
    const restStr = moreMatch[2];

    for (const feat of Object.keys(featureCounts)) {
      if (restStr === feat + "s") {
        return featureCounts[feat] > n;
      }
    }
  }

  const exactMatch = eventStr.match(/^Exactly (\d+) (.+)$/);
  if (exactMatch) {
    const n = parseInt(exactMatch[1], 10);
    const restStr = exactMatch[2];

    const suffix = n > 1 ? "s" : "";
    for (const feat of Object.keys(featureCounts)) {
      if (restStr === feat + suffix) {
        return featureCounts[feat] === n;
      }
    }
  }

  return false;
}

/**
 * Evaluates all events on a bingo card against a specific image.
 *
 * @param {object} imageData - Image data from dataset.json: {id, filename, features: {...}}.
 * @param {object} card - Bingo card: {card_id, events: string[]}.
 * @returns {boolean[]} One boolean per card event; true means the square should be marked.
 */
function evaluateImageForCard(imageData, card) {
  return card.events.map(eventStr => parseEvent(eventStr, imageData.features));
}

/**
 * Checks whether all squares on a card are marked.
 *
 * @param {boolean[]} markings - Array of marked flags.
 * @returns {boolean} True if every square is marked.
 */
function checkCardBingo(markings) {
  return markings.every(Boolean);
}
