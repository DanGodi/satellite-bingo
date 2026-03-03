/**
 * Peer-to-Peer Connection Manager using PeerJS
 *
 * Handles WebRTC P2P connections between host and players.
 * Uses PeerJS for simplified WebRTC signaling.
 *
 * Usage:
 *   const peer = new PeerManager();
 *
 *   // Host creates a game
 *   const gameCode = peer.createHostGame();
 *
 *   // Player joins with code
 *   peer.joinGame(gameCode, onStateUpdate);
 */

class PeerManager {
  constructor() {
    this.peer = null;
    this.hostConnection = null;
    this.playerConnections = new Map(); // peerId -> connection
    this.gameCode = null;
    this.isHost = false;
    this.messageHandlers = [];
  }

  /**
   * Generate a short, memorable game code (4-6 characters).
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
   * Initialize the PeerJS peer instance.
   */
  initializePeer() {
    return new Promise((resolve, reject) => {
      if (this.peer) {
        console.log('Peer already initialized');
        resolve();
        return;
      }

      // Generate a unique peer ID for this browser instance
      const peerId = `${this.generateGameCode()}-${Date.now()}`;

      console.log('Initializing PeerJS with ID:', peerId);

      try {
        this.peer = new Peer(peerId, {
          debug: 1,
          config: {
            iceServers: [
              { urls: 'stun:stun.l.google.com:19302' },
              { urls: 'stun:stun1.l.google.com:19302' },
              // TURN relay servers — fallback when direct P2P fails (symmetric NAT, strict firewalls)
              { urls: 'turn:openrelay.metered.ca:80', username: 'openrelayproject', credential: 'openrelayproject' },
              { urls: 'turn:openrelay.metered.ca:443', username: 'openrelayproject', credential: 'openrelayproject' },
              { urls: 'turn:openrelay.metered.ca:443?transport=tcp', username: 'openrelayproject', credential: 'openrelayproject' }
            ]
          }
        });

        this.peer.on('error', (err) => {
          console.error('PeerJS error:', err.type, err.message);
          reject(err);
        });

        this.peer.on('connection', (conn) => {
          console.log('Incoming connection from:', conn.peer);
          this.handleIncomingConnection(conn);
        });

        // Wait for peer to be ready
        this.peer.on('open', () => {
          console.log('✓ Peer initialized successfully with ID:', peerId);
          resolve();
        });

        // Timeout if peer doesn't initialize
        setTimeout(() => {
          if (!this.peer?.id) {
            reject(new Error('PeerJS initialization timeout - could not reach signaling server'));
          }
        }, 10000);
      } catch (error) {
        console.error('Failed to create Peer instance:', error);
        reject(error);
      }
    });
  }

  /**
   * Host creates a game and waits for players to join.
   */
  async createHostGame() {
    await this.initializePeer();

    this.isHost = true;
    this.gameCode = this.generateGameCode();

    console.log('Host game created with code:', this.gameCode);
    console.log('Host peer ID:', this.peer.id);

    return {
      gameCode: this.gameCode,
      peerId: this.peer.id
    };
  }

  /**
   * Player joins a game using the host's game code and peer ID.
   * Note: In practice, we'll need to get the host's peer ID from the code somehow.
   * For local testing, we can pass it directly. For production, use a signaling service.
   */
  async joinGame(gameCode, hostPeerId) {
    await this.initializePeer();

    this.gameCode = gameCode;

    console.log('Player joining game:', gameCode);
    console.log('Connecting to host peer ID:', hostPeerId);

    // Create connection to host
    this.hostConnection = this.peer.connect(hostPeerId, {
      reliable: true,
      serialization: 'json'
    });

    return new Promise((resolve, reject) => {
      let settled = false;
      const settle = (fn, val) => {
        if (!settled) { settled = true; fn(val); }
      };

      // peer-unavailable fires immediately when the host peer ID doesn't exist
      const onPeerError = (err) => {
        if (err.type === 'peer-unavailable') {
          settle(reject, new Error('Host not found. Make sure the host is on host.html and the Peer ID is correct.'));
        }
      };
      this.peer.on('error', onPeerError);

      this.hostConnection.on('open', () => {
        this.peer.off('error', onPeerError);
        console.log('Connected to host');
        settle(resolve, undefined);
      });

      this.hostConnection.on('error', (err) => {
        console.error('Connection error:', err);
        settle(reject, err);
      });

      this.hostConnection.on('data', (data) => {
        this.handleMessage(data);
      });

      // Timeout after 15 seconds (NAT traversal can be slow)
      setTimeout(() => {
        settle(reject, new Error('Connection timeout - could not reach host'));
      }, 15000);
    });
  }

  /**
   * Host handles incoming player connections.
   */
  handleIncomingConnection(conn) {
    console.log('Player connecting:', conn.peer);

    conn.on('open', () => {
      this.playerConnections.set(conn.peer, conn);
      console.log('Player connected:', conn.peer);

      // Notify listeners of new player
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
      console.log('Player disconnected:', conn.peer);
    });

    conn.on('error', (err) => {
      console.error('Connection error with player:', err);
      this.playerConnections.delete(conn.peer);
    });
  }

  /**
   * Handle incoming messages from peers.
   */
  handleMessage(data, fromPeerId = null) {
    // Notify all registered handlers
    this.messageHandlers.forEach(handler => {
      handler(data, fromPeerId);
    });
  }

  /**
   * Register a callback to handle incoming messages.
   */
  onMessage(callback) {
    this.messageHandlers.push(callback);
  }

  /**
   * Send a message to all connected peers.
   */
  broadcastMessage(data) {
    if (this.isHost) {
      // Host sends to all players
      this.playerConnections.forEach((conn) => {
        if (conn.open) {
          conn.send(data);
        }
      });
    } else {
      // Player sends to host
      if (this.hostConnection && this.hostConnection.open) {
        this.hostConnection.send(data);
      }
    }
  }

  /**
   * Disconnect all connections.
   */
  disconnect() {
    if (this.isHost) {
      this.playerConnections.forEach((conn) => {
        conn.close();
      });
      this.playerConnections.clear();
    } else {
      if (this.hostConnection) {
        this.hostConnection.close();
        this.hostConnection = null;
      }
    }
  }

  /**
   * Get the current game code.
   */
  getGameCode() {
    return this.gameCode;
  }

  /**
   * Get number of connected players (for host only).
   */
  getConnectedPlayerCount() {
    return this.playerConnections.size;
  }
}
