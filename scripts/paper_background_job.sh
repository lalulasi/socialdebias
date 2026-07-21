#!/usr/bin/env bash
# Start/status/log a fixed paper-v2 runner without relying on terminal variables.
set -euo pipefail

project_root=/autodl-fs/data/socialdebias
batch_root=/autodl-fs/data/socialdebias/results/paper_v2_20260719

usage() {
  echo "Usage:"
  echo "  bash scripts/paper_background_job.sh start <job> <runner>"
  echo "  bash scripts/paper_background_job.sh status <job>"
  echo "  bash scripts/paper_background_job.sh log <job>"
}

action="${1:-}"
job="${2:-}"
if [[ ! "${job}" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
  usage
  exit 2
fi

mkdir -p "${batch_root}/logs" "${batch_root}/pids"
pid_file="${batch_root}/pids/${job}.pid"
exit_file="${batch_root}/pids/${job}.exit"
log_file="${batch_root}/logs/${job}_master.log"
legacy_pid_file=""
legacy_runner_name=""
legacy_log_file=""
case "${job}" in
  p7_ablation)
    legacy_pid_file="${batch_root}/p7_resume.pid"
    legacy_runner_name=run_p7_ablation_resume.sh
    legacy_log_file="${batch_root}/logs/p7_resume_master.log"
    ;;
  p8_surface17)
    legacy_pid_file="${batch_root}/p8_resume.pid"
    legacy_runner_name=run_p8_surface17_resume.sh
    legacy_log_file="${batch_root}/logs/p8_resume_master.log"
    ;;
esac

legacy_is_running() {
  [[ -n "${legacy_pid_file}" && -s "${legacy_pid_file}" ]] || return 1
  legacy_pid="$(cat "${legacy_pid_file}")"
  kill -0 "${legacy_pid}" 2>/dev/null || return 1
  legacy_command="$(ps -p "${legacy_pid}" -o args= 2>/dev/null || true)"
  [[ "${legacy_command}" == *"${legacy_runner_name}"* ]]
}

case "${action}" in
  start)
    runner="${3:-}"
    if [[ -z "${runner}" ]]; then
      usage
      exit 2
    fi
    if [[ "${runner}" != /* ]]; then
      runner="${project_root}/${runner}"
    fi
    [[ -f "${runner}" ]] || { echo "[ERROR] Runner not found: ${runner}" >&2; exit 1; }

    if legacy_is_running; then
      echo "[RUNNING/LEGACY] ${job} pid=$(cat "${legacy_pid_file}")"
      echo "Do not start a duplicate; let the existing task finish."
      exit 0
    fi

    if [[ -s "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
      running_command="$(ps -p "$(cat "${pid_file}")" -o args= 2>/dev/null || true)"
      if [[ "${running_command}" == *"${runner}"* ]]; then
        echo "[RUNNING] ${job} pid=$(cat "${pid_file}")"
        echo "log=${log_file}"
        exit 0
      fi
      echo "[WARN] Stale PID file points to another process; replacing it." >&2
    fi

    rm -f "${exit_file}"
    nohup bash -c '
      runner=$1
      exit_file=$2
      bash "${runner}"
      rc=$?
      printf "%s\n" "${rc}" > "${exit_file}"
      exit "${rc}"
    ' paper-background-wrapper "${runner}" "${exit_file}" \
      >> "${log_file}" 2>&1 < /dev/null &
    echo $! > "${pid_file}"
    echo "[STARTED] ${job} pid=$(cat "${pid_file}")"
    echo "log=${log_file}"
    ;;
  status)
    if legacy_is_running; then
      echo "[RUNNING/LEGACY] ${job}"
      ps -p "$(cat "${legacy_pid_file}")" -o pid,etime,args
    elif [[ -s "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
      echo "[RUNNING] ${job}"
      ps -p "$(cat "${pid_file}")" -o pid,etime,args
    else
      if [[ -s "${exit_file}" ]]; then
        rc="$(cat "${exit_file}")"
        if [[ "${rc}" == 0 ]]; then
          echo "[SUCCEEDED] ${job} exit=0"
        else
          echo "[FAILED] ${job} exit=${rc}"
        fi
      else
        echo "[STOPPED/UNKNOWN] ${job}: no exit record"
      fi
      tail -n 20 "${log_file}" 2>/dev/null || true
    fi
    ;;
  log)
    if legacy_is_running && [[ -n "${legacy_log_file}" ]]; then
      touch "${legacy_log_file}"
      tail -f "${legacy_log_file}"
      exit 0
    fi
    touch "${log_file}"
    tail -f "${log_file}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
