hi

docker build --platform linux/amd64 -t x86-env . && docker run --platform linux/amd64 -it -v $(pwd)/src:/src x86-env