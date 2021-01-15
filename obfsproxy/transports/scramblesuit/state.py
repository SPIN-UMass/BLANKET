"""
Provide a way to store the server's state information on disk.

The server possesses state information which should persist across runs.  This
includes key material to encrypt and authenticate session tickets, replay
tables and PRNG seeds.  This module provides methods to load, store and
generate such state information.
"""

import os
import sys
import time
import cPickle
import random

import const
import replay
import mycrypto
import probdist

import obfsproxy.common.log as logging

log = logging.get_obfslogger()

def load( ):
    """
    Load the server's state object from file.

    The server's state file is loaded and the state object returned.  If no
    state file is found, a new one is created and returned.
    """

    stateFile = const.STATE_LOCATION + const.SERVER_STATE_FILE

    log.info("Attempting to load the server's state file from `%s'." %
             stateFile)

    if not os.path.exists(stateFile):
        log.info("The server's state file does not exist (yet).")
        state = State()
        state.genState()
        return state

    try:
        with open(stateFile, 'r') as fd:
            stateObject = cPickle.load(fd)
    except IOError as err:
        log.error("Error reading server state file from `%s': %s" %
                  (stateFile, err))
        sys.exit(1)

    return stateObject

class State( object ):

    """
    Implement a state class which stores the server's state.

    This class makes it possible to store state information on disk.  It
    provides methods to generate and write state information.
    """

    def __init__( self ):
        """
        Initialise a `State' object.
        """

        self.prngSeed = None
        self.keyCreation = None
        self.hmacKey = None
        self.aesKey = None
        self.oldHmacKey = None
        self.oldAesKey = None
        self.ticketReplay = None
        self.uniformDhReplay = None
        self.pktDist = None
        self.iatDist = None
        self.fallbackPassword = None

    def genState( self ):
        """
        Populate all the local variables with values.
        """

        log.info("Generating parameters for the server's state file.")

        # PRNG seed for the client to reproduce the packet and IAT morpher.
        self.prngSeed = mycrypto.strongRandom(const.PRNG_SEED_LENGTH)

        # HMAC and AES key used to encrypt and authenticate tickets.
        self.hmacKey = mycrypto.strongRandom(const.TICKET_HMAC_KEY_LENGTH)
        self.aesKey = mycrypto.strongRandom(const.TICKET_AES_KEY_LENGTH)
        self.keyCreation = int(time.time())

        # The previous HMAC and AES keys.
        self.oldHmacKey = None
        self.oldAesKey = None

        # Replay dictionary for both authentication mechanisms.
        self.replayTracker = replay.Tracker()

        # Distributions for packet lengths and inter arrival times.
        prng = random.Random(self.prngSeed)
        self.pktDist = probdist.new(lambda: prng.randint(const.HDR_LENGTH,
                                                         const.MTU),
                                    seed=self.prngSeed)
        self.iatDist = probdist.new(lambda: prng.random() %
                                    const.MAX_PACKET_DELAY,
                                    seed=self.prngSeed)

        # Fallback UniformDH shared secret.  Only used if the bridge operator
        # did not set `ServerTransportOptions'.
        self.fallbackPassword = os.urandom(const.SHARED_SECRET_LENGTH)

        self.writeState()

    def isReplayed( self, hmac ):
        """
        Check if `hmac' is present in the replay table.

        Return `True' if the given `hmac' is present in the replay table and
        `False' otherwise.
        """

        assert self.replayTracker is not None

        log.debug("Querying if HMAC is present in the replay table.")

        return self.replayTracker.isPresent(hmac)

    def registerKey( self, hmac ):
        """
        Add the given `hmac' to the replay table.
        """

        assert self.replayTracker is not None

        log.debug("Adding a new HMAC to the replay table.")
        self.replayTracker.addElement(hmac)

        # We must write the data to disk immediately so that other ScrambleSuit
        # connections can share the same state.
        self.writeState()

    def writeState( self ):
        """
        Write the state object to a file using the `cPickle' module.
        """

        stateFile = const.STATE_LOCATION + const.SERVER_STATE_FILE

        log.debug("Writing server's state file to `%s'." %
                  stateFile)

        try:
            with open(stateFile, 'w') as fd:
                cPickle.dump(self, fd)
        except IOError as err:
            log.error("Error writing state file to `%s': %s" %
                      (stateFile, err))
            sys.exit(1)
