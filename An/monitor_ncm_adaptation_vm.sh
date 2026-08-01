#!/usr/bin/env bash
set -u

PROJECT="project-95a0d104-9d0f-4aa1-ba0"
ZONE="us-central1-b"
INSTANCE="anmetarayban"
REMOTE_ROOT="/home/an/RaybanMeta/uav-continual-learning"
POLL_SECONDS=300

describe_status() {
  gcloud compute instances describe "$INSTANCE" \
    --zone="$ZONE" --project="$PROJECT" --format='value(status)' 2>/dev/null
}

remote_state() {
  gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
    --command="cd '$REMOTE_ROOT' && if [ -f artifacts_ncm_adaptation/REPORT_TITANS_NCM_NO_REPLAY.md ]; then echo COMPLETE; elif pgrep -f '[r]un_ncm_adaptation_study.py' >/dev/null; then echo ACTIVE; else echo IDLE; fi" \
    2>/dev/null | tail -n 1
}

resume_campaign() {
  gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
    --command="cd '$REMOTE_ROOT' && (tmux has-session -t ncm_adaptation 2>/dev/null || tmux new-session -d -s ncm_adaptation \"bash -lc 'python3 scripts/run_ncm_adaptation_study.py all --artifact-root artifacts_ncm_adaptation --shutdown >> ncm_adaptation_campaign.log 2>&1'\")"
}

wait_for_ssh() {
  local attempt
  for attempt in $(seq 1 12); do
    if gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" \
      --command="true" >/dev/null 2>&1; then
      return 0
    fi
    sleep 15
  done
  return 1
}

while true; do
  now=$(date '+%Y-%m-%d %H:%M:%S')
  status=$(describe_status)
  echo "[$now] VM=$status"

  if [ "$status" = "RUNNING" ]; then
    state=$(remote_state)
    echo "[$now] campaign=$state"
    if [ "$state" = "COMPLETE" ]; then
      exit 0
    fi
    if [ "$state" = "IDLE" ]; then
      resume_campaign || true
    fi
  elif [ "$status" = "TERMINATED" ]; then
    if gcloud compute instances start "$INSTANCE" \
      --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1; then
      if wait_for_ssh; then
        state=$(remote_state)
        echo "[$now] campaign-after-start=$state"
        if [ "$state" = "COMPLETE" ]; then
          exit 0
        fi
        resume_campaign || true
      fi
    fi
  fi

  sleep "$POLL_SECONDS"
done
