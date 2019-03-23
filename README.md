# nanyang-code-farmer

#### Sample Command for preprocess.py
	python -m preprocess -f data/train_test/*


#### Sample Command for beauty_text.py, fashion_text.py, mobile_text.py

	python -m beauty_text \
	-i processed_data/combined/beauty_train_processed.csv \
	-t processed_data/combined/beauty_val_processed.csv \
	-m data/beauty_profile_train.json

	python -m fashion_text \
	-i processed_data/combined/fashion_train_processed.csv \
	-t processed_data/combined/fashion_val_processed.csv \
	-m data/fashion_profile_train.json

	python -m mobile_text \
	-i processed_data/combined/mobile_train_processed.csv \
	-t processed_data/combined/mobile_val_processed.csv \
	-m data/mobile_profile_train.json

#### Sample Command for combine_pred.py
	python -m combine_pred -f data/*proba.csv
A *submission.csv* will be found in the data/ folder.
