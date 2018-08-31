# Run worker after setting up environment (typically for condor)

if [ "$2" != "" ]; then
  source $HOME/.virtualenvs/$2/bin/activate
fi
python3 -m mihifepe.worker $1
