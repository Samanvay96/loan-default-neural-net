.PHONY: build shell

# Call with `make build test` or `make test`

build:
	docker build -f ops/Dockerfile -t neural-sat .

shell:
	docker run -it \
		-v $(shell pwd):/home/python \
		neural-sat \
		bash
