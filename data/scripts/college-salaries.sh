#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
kaggle datasets download wsj/college-salaries -p $SCRIPT_DIR/../raw --unzip