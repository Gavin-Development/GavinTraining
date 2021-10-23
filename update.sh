echo "Updating code base."
git fetch origin --prune
git pull https://github.com/Gavin-Development/GavinTraining master
git submodule init
git submodule update
echo "Code base up to date."
echo "Updating requirements"
pip3 install -r requirements.txt
cd GavinBackend && pip3 install -r requirements.txt && git pull https://github.com/Gavin-Development/GavinBackend main || exit
echo "Requirements up to date."