# FlowProject - Reward Sharing for multiagent traffic light control
## Instalation Instructions
* Run git clone --recurse-submodules https://www.github.com/.../FlowProject
* Pull branch 780 for flow (this is a pull request that fixes an issue with the traffic light domain 
* if you are installing on your own computer, run flow/scripts/setup_sumo_ubuntuX.sh where X is your ubuntu version
* If you are installing on lab machine, run setup_sumo_ubuntu1804_noDependencies.sh
* next create a virtualenv
python3 -m virtualenv env
source env/bin/activate
* next intsall flow
cd flow
pip install -e ./
* next install other dependencies
pip install -r requirements.txt
* Now run run_experiment.py
python run_experiment.py
