.PHONY: help install test start stop

help:
	@echo "Targets: install | test | start | stop"

install:
	pip install -r requirements.txt

test:
	pytest

start:
	./start.sh

stop:
	./stop.sh
