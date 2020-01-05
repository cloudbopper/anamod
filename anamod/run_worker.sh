# Run worker after setting up environment (typically for condor)

if [ "$2" != "" ]; then
  source $2/bin/activate
fi
python3 -m anamod.worker $1
