# nanyang-code-farmer

#### Sample Command for preprocess.py
	python -m preprocess -f data/train_test/*


#### Sample Command for _text.py

	python -m mobile_text -i data/mobile_data_info_train_competition.csv -t data/mobile_data_info_val_competition.csv -m data/mobile_profile_train.json

#### Sample Command for combine_pred.py
	python -m combine_pred -f data/output/*

#### Sample Command for ocr_mp.py

E.g. The `mobile_image` folder containing the images is in the folder `img`. Allocated for row 10 to row 19 (inclusive)

```.sh
python -m ocr_mp -f beauty_data_info_train_competition_ocr.csv -i img/ -s 10 -e 19
```

#### Allocation of rows

* Huixian: `-e 39999`
* Ziqing: `-s 40000 -e 79999`
* Shande:  `-s 80000 -e 119999`
* Zhiwei:  `-s 120000`
