#!/bin/bash
echo "*** Compile code ***"
python setup_p2p.py build
echo "*** install code ***"
python setup_p2p.py install --user
