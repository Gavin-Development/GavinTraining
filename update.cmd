echo "Updating code base."
git fetch origin --prune
git submodule init
git submodule update
echo "Code base up to date."
echo "Updating requirements"
pip3 install -r requirements.txt
cd GavinBackend && pip3 install -r requirements.txt
echo "Requirements up to date."