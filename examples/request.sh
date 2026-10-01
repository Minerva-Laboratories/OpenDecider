#!/usr/bin/env bash
# Call a running server (python -m opendecider.serve). Prints the JSON response.
curl -s localhost:${PORT:-8000}/v1/decide -H 'content-type: application/json' -d '{
  "state": {"ticket": "I was charged twice for my March invoice and want my money back."},
  "questions": {
    "route":  {"type": "choice", "prompt": "Which team handles this?", "options": ["billing", "technical", "refund", "other"]},
    "urgent": {"type": "noul",   "prompt": "Does this need a reply within the hour?"}
  }
}'
echo
