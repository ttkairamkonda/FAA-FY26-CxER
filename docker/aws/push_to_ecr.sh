#!/usr/bin/env bash
# Build the GPU pipeline image and push it to Amazon ECR, so anyone on the
# team can pull it onto an AWS GPU instance without building it themselves.
#
# Run from anywhere on a machine with Docker + the AWS CLI, logged in with
# permission to create/push ECR repositories:
#
#   bash docker/aws/push_to_ecr.sh                 # uses AWS_REGION or us-east-1
#   AWS_REGION=us-west-2 ECR_REPO=faa-cxer-pipeline TAG=gpu bash docker/aws/push_to_ecr.sh
#
# Prints the image URI at the end; put it in .env as CXER_IMAGE=...

set -euo pipefail

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
REPO="${ECR_REPO:-faa-cxer-pipeline}"
TAG="${TAG:-gpu}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
URI="$REGISTRY/$REPO:$TAG"

echo "Account $ACCOUNT, region $REGION -> $URI"

aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$REGION" --repository-name "$REPO" \
       --image-scanning-configuration scanOnPush=true >/dev/null

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

# EC2 GPU instances are x86_64; build for that even from an Apple Silicon Mac.
docker build --platform linux/amd64 -f "$ROOT/docker/pipeline.Dockerfile" -t "$URI" "$ROOT"
docker push "$URI"

echo
echo "Pushed: $URI"
echo "Teammates run it with:  CXER_IMAGE=$URI bash docker/aws/run_on_ec2.sh all"
