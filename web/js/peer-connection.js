/**
 * Peer-to-Peer Connection Manager using PeerJS.
 *
 * Wraps WebRTC signaling for host↔player communication.
 * The host initializes a named peer; players connect to it by ID.
 *
 * Usage:
 *   const peer = new PeerManager();
 *   await peer.joinGame(hostPeerId, hostPeerId);
 */

class PeerManager {
  constructor() {
    this.peer = null;
    this.hostConnection = null;
    this.playerConnections = new Map();
    this.gameCode = null;
    this.isHost = false;
    this.messageHandlers = [];
  }

  /**
   * Generates a short random alphanumeric code.
   *
   * @returns {string} A 5-character uppercase code.
   */
  generateGameCode() {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let code = '';
    for (let i = 0; i < 5; i++) {
      code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
  }

  /**
   * Initializes the PeerJS peer instance with a unique ID.
   *
   * @returns {Promise<void>} Resolves when the peer is open and ready.
   */
  initializePeer() {
    return new Promise((resolve, reject) => {
      if (this.peer) {
        resolve();
        return;
      }

      const peerId = `${this.generateGameCode()}-${Date.now()}`;

      try {
        this.peer = new Peer(peerId, {
          debug: 1,
          config: {
            iceServers: [
              { urls: 'stun:stun.l.google.com:19302' },
              { urls: 'stun:stun1.l.google.com:19302' }
            ]
          }
        });

        this.peer.on('error', (err) => {
          reject(err);
        });

        this.peer.on('connection', (conn) => {
          this.handleIncomingConnection(conn);
        });

        this.peer.on('open', () => {
          resolve();
        });

        setTimeout(() => {
          if (!this.peer?.open) {
            reject(new Error('PeerJS initialization timeout - could not reach signaling server'));
          }
        }, 10000);
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Connects a player to the host using the host's peer ID.
   *
   * @param {string} gameCode - The game code (same as hostPeerId in current design).
   * @param {string} hostPeerId - The full PeerJS ID of the host peer.
   * @returns {Promise<void>} Resolves when the connection is established.
   */
  async joinGame(gameCode, hostPeerId) {
    await this.initializePeer();

    this.gameCode = gameCode;

    this.hostConnection = this.peer.connect(hostPeerId, {
      reliable: true,
      serialization: 'json'
    });

    return new Promise((resolve, reject) => {
      let settled = false;
      const settle = (fn, val) => {
        if (!settled) { settled = true; fn(val); }
      };

      const onPeerError = (err) => {
        if (err.type === 'peer-unavailable') {
          settle(reject, new Error('Host not found. Make sure the host is on host.html and the Peer ID is correct.'));
        }
      };
      this.peer.on('error', onPeerError);

      this.hostConnection.on('open', () => {
        this.peer.off('error', onPeerError);
        settle(resolve, undefined);
      });

      this.hostConnection.on('error', (err) => {
        settle(reject, err);
      });

      this.hostConnection.on('data', (data) => {
        this.handleMessage(data);
      });

      setTimeout(() => {
        settle(reject, new Error('Connection timeout - could not reach host'));
      }, 15000);
    });
  }

  /**
   * Handles an incoming connection from a new player (host only).
   *
   * @param {DataConnection} conn - The incoming PeerJS connection.
   */
  handleIncomingConnection(conn) {
    conn.on('open', () => {
      this.playerConnections.set(conn.peer, conn);
      this.broadcastMessage({
        type: 'player-joined',
        playerId: conn.peer
      });
    });

    conn.on('data', (data) => {
      this.handleMessage(data, conn.peer);
    });

    conn.on('close', () => {
      this.playerConnections.delete(conn.peer);
    });

    conn.on('error', () => {
      this.playerConnections.delete(conn.peer);
    });
  }

  /**
   * Dispatches an incoming message to all registered handlers.
   *
   * @param {object} data - The message payload.
   * @param {string|null} fromPeerId - Sender's peer ID, or null if from host.
   */
  handleMessage(data, fromPeerId = null) {
    this.messageHandlers.forEach(handler => {
      handler(data, fromPeerId);
    });
  }

  /**
   * Registers a callback to handle incoming messages.
   *
   * @param {function(object, string|null): void} callback - Message handler.
   */
  onMessage(callback) {
    this.messageHandlers.push(callback);
  }

  /**
   * Sends a message to all connected players (host) or to the host (player).
   *
   * @param {object} data - The message payload.
   */
  broadcastMessage(data) {
    if (this.isHost) {
      this.playerConnections.forEach((conn) => {
        if (conn.open) {
          conn.send(data);
        }
      });
    } else {
      if (this.hostConnection && this.hostConnection.open) {
        this.hostConnection.send(data);
      }
    }
  }

  /**
   * Sends a message to the host (player only). Alias for broadcastMessage.
   *
   * @param {object} data - The message payload.
   */
  sendToHost(data) {
    this.broadcastMessage(data);
  }

  /**
   * Sends a message to a specific connected player (host only).
   *
   * @param {string} peerId - The target player's peer ID.
   * @param {object} data - The message payload.
   */
  sendToPlayer(peerId, data) {
    if (!this.isHost) {
      console.warn('sendToPlayer() can only be called by the host');
      return;
    }
    const conn = this.playerConnections.get(peerId);
    if (conn && conn.open) {
      conn.send(data);
    }
  }

  /**
   * Closes all active connections.
   */
  disconnect() {
    if (this.isHost) {
      this.playerConnections.forEach((conn) => conn.close());
      this.playerConnections.clear();
    } else {
      if (this.hostConnection) {
        this.hostConnection.close();
        this.hostConnection = null;
      }
    }
  }

  /**
   * Returns the current game code.
   *
   * @returns {string|null}
   */
  getGameCode() {
    return this.gameCode;
  }

  /**
   * Returns the number of connected players (host only).
   *
   * @returns {number}
   */
  getConnectedPlayerCount() {
    return this.playerConnections.size;
  }
}
