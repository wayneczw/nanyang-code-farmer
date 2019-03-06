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

#### Allocation of rows (Mobile Image OCR)

* Huixian: `-e 39999`
* Ziqing: `-s 40000 -e 79999`
* Shande:  `-s 80000 -e 119999`
* Zhiwei:  `-s 120000`

#### Further Allocation of OCR
* Ziqing: `fashion_image`
* Shande:  `beauty_image`

#### Sample Command for select_by_attributes.py

E.g. select those rows where either colour_group or skin_type is not NA. Rows with all selected attributes being NA will be dropped.
Attribute names are not case sensitive.

```.sh
python -m select_by_attributes -f beauty_data_info_train_competition.csv -a colour_group skin_type
```

If the attribute name has white space, use `""`

```.sh
python -m select_by_attributes -f mobile_data_info_train_competition.csv  -a features "warranty period" "color family" camera "phone screen size"
```
