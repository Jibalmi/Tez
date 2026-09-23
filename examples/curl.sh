#!/usr/bin/env sh
# The docs/API.md request against a local Tez server. Start one first, e.g.
#   tez serve --backend http://127.0.0.1:8091 --template gemma4          # port 8787
# Any bearer token is accepted unless the server was started with --api-key.
set -eu
TEZ="${TEZ:-http://127.0.0.1:8787}"
AUTH="Authorization: Bearer ${TEZ_API_KEY:-local}"

curl -sS "$TEZ/v1/systemone" -H "Content-Type: application/json" -H "$AUTH" -d @- <<'JSON'
{
  "model": "tez-latest",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?",
      "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}
    },
    "topic": {
      "type": "choice",
      "instructions": "What is the message about?",
      "criteria": {"billing": "Payments, payouts, invoices", "technical": "Something is broken", "sales": null}
    },
    "anger": {
      "type": "score",
      "instructions": "How upset is the writer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
JSON
echo

curl -sS "$TEZ/v1/models" -H "$AUTH"; echo
curl -sS "$TEZ/healthz"; echo

# With schemas loaded (tez serve --schemas schemas/), a correct label for a past decision feeds `tez fit`:
# curl -sS "$TEZ/v1/feedback" -H "Content-Type: application/json" -H "$AUTH" \
#   -d '{"schema": "support-triage", "question": "topic", "state": "Help! My payouts have been failing for 3 days.", "label": "billing"}'
