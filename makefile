all: build run

build:
	docker build --platform linux/amd64 -t x86-env .

run:
	docker run --platform linux/amd64 -it -v $(PWD)/src:/src x86-env

prune:
	docker container prune

delete:
	docker rmi -f x86-env