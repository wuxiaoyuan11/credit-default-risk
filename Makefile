.PHONY: setup run dashboard streamlit clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

run:
	.venv/bin/python src/train_credit_default.py

dashboard:
	open dashboard/index.html

streamlit:
	.venv/bin/streamlit run app.py

clean:
	find . -name ".DS_Store" -delete
