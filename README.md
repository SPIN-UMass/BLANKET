# BLANKET


Tor Transport pluggin for the paper: "Defeating DNN-Based Trafﬁc Analysis Systems in Real-Time With Blind Adversarial Perturbations"
https://arxiv.org/pdf/2002.06495




Based on obfs3



Step 0: Install Python

 To use obfsproxy you will need Python (>= 2.7) and pip. If you use
 Debian testing (or unstable), or a version of Ubuntu newer than
 Oneiric, this is easy:

   # apt-get install python2.7 python-pip python-dev build-essential libgmp-dev


Step 1: Install Tor

 You will also need a development version of Tor. To do this, you
 should use the following guide to install tor and
 deb.torproject.org-keyring:
 https://www.torproject.org/docs/debian.html.en#development

 You need Tor 0.2.4.x because it knows how to automatically report
 your obfsproxy address to BridgeDB.


Step 2: Install nnmorph

  If you have pip, installing obfsproxy and its dependencies should be
  a matter of a single command:

    $ python setup.py install


Step 3: Setup Tor

  Now setup Tor. Edit your /etc/tor/torrc to add:

    SocksPort 0
    ORPort 443 # or some other port if you already run a webserver/skype
    BridgeRelay 1
    Exitpolicy reject *:*

    ## CHANGEME_1 -> provide a nickname for your bridge, can be anything you like
    #Nickname CHANGEME_1
    ## CHANGEME_2 -> provide some email address so we can contact you if there's a problem
    #ContactInfo CHANGEME_2

    ServerTransportPlugin nnmorph exec /usr/local/bin/obfsproxy managed

