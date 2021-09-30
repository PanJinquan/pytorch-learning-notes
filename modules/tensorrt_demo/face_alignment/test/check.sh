#!/usr/bin/env bash
LOG_PATH=${1}

echo '[WorkDir]:'$(pwd)>>${LOG_PATH}

# run your test files
python check_checkpoint.py ${LOG_PATH}
FLAG=$?
if [ $FLAG -eq 1 ];then
  exit 1
fi


exit 0


