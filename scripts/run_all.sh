#! /bin/bash

rye run python src/safety_project/create_user_interactions_from_squad.py
rye run python src/safety_project/eval_simple_1000.py
rye run python src/safety_project/create_chart.py

