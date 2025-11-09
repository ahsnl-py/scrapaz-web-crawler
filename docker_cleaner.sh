#!/usr/bin/env bash
set -euo pipefail

echo "Removing stopped containers..."
docker container prune -f

echo "Removing dangling images..."
docker image prune -f

echo "Removing unused networks..."
docker network prune -f

echo "Removing unused volumes..."
docker volume prune -f

echo "Removing all unused images (be careful: this includes cached layers)..."
docker image prune -a -f

echo "Docker cleanup complete."