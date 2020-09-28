build:
	docker build . -t test_split

run:
	docker run --rm -p 8000:8000 test_split

main: build run