#!/usr/bin/env bash
set -euo pipefail

has_command() {
  command -v "$1" >/dev/null 2>&1
}

run_as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
    return
  fi

  if has_command sudo; then
    sudo "$@"
    return
  fi

  printf 'Redis is not installed and sudo is unavailable.\n' >&2
  exit 1
}

install_redis() {
  if has_command redis-server; then
    return
  fi

  printf 'redis-server was not found. Installing it now...\n'

  if has_command apt-get; then
    run_as_root apt-get update
    run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y redis-server
    return
  fi

  if has_command dnf; then
    run_as_root dnf install -y redis
    return
  fi

  if has_command yum; then
    run_as_root yum install -y redis
    return
  fi

  if has_command pacman; then
    run_as_root pacman -Syu --noconfirm redis
    return
  fi

  if has_command zypper; then
    run_as_root zypper --non-interactive install redis
    return
  fi

  if has_command apk; then
    run_as_root apk add redis
    return
  fi

  printf 'Could not find a supported package manager to install Redis.\n' >&2
  exit 1
}

redis_is_running() {
  if ! has_command redis-cli; then
    return 1
  fi

  local host="${REDIS_HOST:-127.0.0.1}"
  local port="${REDIS_PORT:-6379}"

  redis-cli -h "$host" -p "$port" ping >/dev/null 2>&1
}

main() {
  install_redis

  if redis_is_running; then
    printf 'Redis is already running on %s:%s; not starting another redis-server.\n' \
      "${REDIS_HOST:-127.0.0.1}" \
      "${REDIS_PORT:-6379}"
    exit 0
  fi

  exec "$(command -v redis-server)" "$@"
}

main "$@"