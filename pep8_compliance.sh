#!/bin/bash

find . -name '*.py' -exec pycodestyle {} \; > pep8_report.txt
