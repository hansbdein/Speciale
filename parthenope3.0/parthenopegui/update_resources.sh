#!/bin/bash
pyrcc5 -o resourcesPySide2.py resources.qrc
sed -i s#PyQt5#PySide2#g resourcesPySide2.py
