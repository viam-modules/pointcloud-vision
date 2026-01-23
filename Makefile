setup:
	./setup.sh

build:
	./build.sh

test:
	pytest -v tests/

lint:
	black .