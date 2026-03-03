/**
 * Game state management for Satellite Bingo.
 *
 * Handles:
 *  - Loading dataset.json and cards.json
 *  - Shuffling images
 *  - Tracking game progression (current turn, marked squares per card)
 *  - Auto-evaluating events when images advance
 *  - Detecting bingo winners
 *  - Persisting state to localStorage
 */

/**
 * Main game state class.
 *
 * @property {object[]} allImages - Array of image objects from dataset.json
 * @property {object[]} cards - Array of bingo card objects from cards.json
 * @property {number} currentTurn - Current turn index (-1 means game not started)
 * @property {object} cardMarkings - Map of cardId -> boolean[] (marked squares per event)
 * @property {string} gameId - Unique ID for this game session (stored in localStorage)
 */
class GameState {
  constructor(dataset, cards, gameId = null) {
    this.allImages = [...dataset.images];
    this.cards = cards;
    this.currentTurn = -1;
    this.cardMarkings = {};
    this.gameId = gameId || this.generateGameId();

    // Initialize markings: each card gets an array of false (unmarked) for each event
    for (const card of cards) {
      this.cardMarkings[card.card_id] = new Array(card.events.length).fill(false);
    }

    this.shuffle();
    this.loadFromStorage();
  }

  /**
   * Generate a unique game ID.
   * @returns {string}
   */
  generateGameId() {
    return `bingo_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  }

  /**
   * Fisher-Yates shuffle of images.
   */
  shuffle() {
    for (let i = this.allImages.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.allImages[i], this.allImages[j]] = [this.allImages[j], this.allImages[i]];
    }
  }

  /**
   * Advance to the next image and auto-evaluate all events.
   *
   * @returns {object|null} The image data for the current turn, or null if game over
   */
  advanceTurn() {
    this.currentTurn++;
    if (this.currentTurn >= this.allImages.length) {
      this.currentTurn = this.allImages.length - 1;
      return null;
    }

    const currentImage = this.allImages[this.currentTurn];

    // Auto-evaluate all events for all cards
    for (const card of this.cards) {
      const fires = evaluateImageForCard(currentImage, card);
      for (let i = 0; i < fires.length; i++) {
        if (fires[i]) {
          this.cardMarkings[card.card_id][i] = true;
        }
      }
    }

    this.saveToStorage();
    return currentImage;
  }

  /**
   * Manually mark/unmark a square on a card (for player manual mode).
   *
   * @param {number} cardId
   * @param {number} eventIndex - Index into the card's events array
   * @param {boolean} marked
   */
  setSquareMarked(cardId, eventIndex, marked) {
    if (this.cardMarkings[cardId]) {
      this.cardMarkings[cardId][eventIndex] = marked;
      this.saveToStorage();
    }
  }

  /**
   * Check if a card has won (all squares marked).
   *
   * @param {number} cardId
   * @returns {boolean}
   */
  checkBingo(cardId) {
    if (!this.cardMarkings[cardId]) return false;
    return this.cardMarkings[cardId].every(Boolean);
  }

  /**
   * Get all cards that have won, in order of winning.
   *
   * @returns {array} Array of {cardId, turn, position} objects
   */
  getWinners() {
    const winners = [];
    for (const card of this.cards) {
      if (this.checkBingo(card.card_id)) {
        winners.push({
          cardId: card.card_id,
          turn: this.currentTurn,
        });
      }
    }
    return winners;
  }

  /**
   * Get current image (or null if game not started).
   * @returns {object|null}
   */
  getCurrentImage() {
    if (this.currentTurn < 0 || this.currentTurn >= this.allImages.length) {
      return null;
    }
    return this.allImages[this.currentTurn];
  }

  /**
   * Check which cards have marked at least N squares.
   *
   * @param {number} minMarked - Minimum number of marked squares
   * @returns {number[]} Array of card IDs
   */
  getCardsWithProgress(minMarked) {
    const result = [];
    for (const card of this.cards) {
      const marked = this.cardMarkings[card.card_id].filter(Boolean).length;
      if (marked >= minMarked) {
        result.push(card.card_id);
      }
    }
    return result;
  }

  /**
   * Save game state to localStorage.
   */
  saveToStorage() {
    const state = {
      gameId: this.gameId,
      currentTurn: this.currentTurn,
      shuffledImages: this.allImages.map(img => img.id),
      cardMarkings: this.cardMarkings,
    };
    localStorage.setItem(`bingo_game_${this.gameId}`, JSON.stringify(state));
  }

  /**
   * Load game state from localStorage if it exists.
   *
   * @returns {boolean} True if state was loaded from storage
   */
  loadFromStorage() {
    const stored = localStorage.getItem(`bingo_game_${this.gameId}`);
    if (!stored) return false;

    try {
      const state = JSON.parse(stored);
      this.currentTurn = state.currentTurn;
      this.cardMarkings = state.cardMarkings;
      // Note: we don't restore shuffled images - that would require re-shuffling the array
      // Instead, we keep the current shuffle and just restore the turn and markings
      return true;
    } catch (e) {
      console.error('Failed to load game state from localStorage:', e);
      return false;
    }
  }

  /**
   * Clear game state from localStorage.
   */
  clearStorage() {
    localStorage.removeItem(`bingo_game_${this.gameId}`);
  }
}

/**
 * Load dataset.json from the web/data directory.
 *
 * @returns {Promise<object>} The parsed dataset
 */
async function loadDataset() {
  const response = await fetch('data/dataset.json');
  if (!response.ok) {
    throw new Error(`Failed to load dataset.json: ${response.statusText}`);
  }
  return await response.json();
}

/**
 * Load cards.json from the web/data directory.
 *
 * @returns {Promise<array>} Array of bingo card objects
 */
async function loadCards() {
  const response = await fetch('data/cards.json');
  if (!response.ok) {
    throw new Error(`Failed to load cards.json: ${response.statusText}`);
  }
  return await response.json();
}
