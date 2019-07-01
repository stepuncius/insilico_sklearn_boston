#!/usr/bin/env bash
curl -X post -H "Content-Type: application/json" \
 -d '{"CRIM": 0.62739, "ZN": 0.0, "INDUS": 8.14, "CHAS": 0.0, "NOX": 0.538, "RM": 5.834, "AGE": 56.5, "DIS": 4.4986, "RAD": 4.0, "TAX": 307.0, "PTRATIO": 21.0, "B": 395.62, "LSTAT": 8.47}' \
 localhost:5000/predict
