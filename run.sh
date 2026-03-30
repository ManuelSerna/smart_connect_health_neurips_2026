train () {
  local args=("$@")
  #local param1="${args[0]}"
  #local param2="${args[1]}"

  python train.py
}

evaluate () {
  local args=("$@")
  local task="${args[0]}"
  #local param2="${args[1]}"

  python eval.py --task "${task}"
}

# Run a program
choice="${1}"
task="${2}"

if [[ ${choice} == "train" ]]; then
  echo "Choice: Train"
  train
elif [[ ${choice} == "eval" ]]; then
  echo "Choice: Evaluate"
  evaluate "${task}"
else
  echo "[ERROR] Invalid choice ('${choice}') in 'run.sh'."
  exit 1
fi
