# Run worker after setting up environment (typically for condor)

if [ "$2" != "" ]; then
  $HOME/.virtualenvs/$2/bin/activate_this.py
fi
python3 -m mihifepe.worker $1
