.PHONY: test

test:
	cd test && dub build --force --compiler=$(DC) && python test.py
