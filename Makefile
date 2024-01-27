 requirements:
	conda env export > environment.yml

setup:
	conda env create -f environment.yml

train:
	python -m procodex.train

inference:
	python3 -m procodex.inference
