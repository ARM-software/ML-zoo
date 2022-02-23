python3 -m venv env
source env/bin/activate

pip install -r requirements.txt
python preprocessing.py
python train_model.py --with_baseline --baseline_epochs 30 --with_finetuning --finetuning_epochs 10 --with_fluent_speech --fluent_speech_epochs 30
python prune_and_quantise_model.py --prune --sparsity 0.5 --finetuning_epochs 10
python prune_and_quantise_model.py --sparsity 0.5 --finetuning_epochs 10



