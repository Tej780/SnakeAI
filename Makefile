requirqments:
	@echo 'Install python3 requirements into linux machine'
	sudo apt-get install python3-pip python3-venv
	sudo pip3 install virtualenv

env:
	@echo 'Install requirements into virtualenv'
	python3 -m venv .venv
	.venv/bim/pip3 install -r requirements.txt
	
run:
	@echo 'Run Snake Deep Q Network'
	python3 snake-V0-DQN.py

