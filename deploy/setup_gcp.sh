#!/usr/bin/env bash
# One-time Google Cloud setup for the Verbatim batch job. Safe to re-run.
#
#   read -rs GROQ_API_KEY && export GROQ_API_KEY   # needed the first time, never echoed
#   bash deploy/setup_gcp.sh <project-id> [region]
#
# Creates an Artifact Registry repository, a data bucket, a least-privilege
# service account for the job, and lets Cloud Build deploy the job.
set -euo pipefail

PROJECT_ID="${1:?usage: setup_gcp.sh <project-id> [region]}"
REGION="${2:-europe-west2}"
REPOSITORY="verbatim"
BUCKET="${PROJECT_ID}-verbatim-data"
JOB_SA="verbatim-batch@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
# Cloud Build runs as the Compute Engine default service account in new projects.
BUILD_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "==> Enabling APIs"
# compute.googleapis.com creates the default service account Cloud Build runs as.
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com aiplatform.googleapis.com \
  secretmanager.googleapis.com storage.googleapis.com compute.googleapis.com

echo "==> Artifact Registry repository: $REPOSITORY"
gcloud artifacts repositories describe "$REPOSITORY" --location="$REGION" >/dev/null 2>&1 ||
  gcloud artifacts repositories create "$REPOSITORY" --repository-format=docker \
    --location="$REGION" --description="Verbatim container images"

echo "==> Data bucket: gs://$BUCKET"
gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 ||
  gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access

echo "==> Job service account: $JOB_SA"
if ! gcloud iam service-accounts describe "$JOB_SA" >/dev/null 2>&1; then
  gcloud iam service-accounts create verbatim-batch --display-name="Verbatim batch job"
  sleep 15  # a new service account takes a moment before IAM grants accept it
fi
# Only what the job needs: call Vertex AI and read/write its own bucket.
gcloud projects add-iam-policy-binding "$PROJECT_ID" --condition=None --quiet \
  --member="serviceAccount:$JOB_SA" --role=roles/aiplatform.user >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:$JOB_SA" --role=roles/storage.objectUser >/dev/null

echo "==> Cloud Build permissions: $BUILD_SA"
# Granted explicitly: organisation policies often withhold Editor from this account.
# storage.objectViewer lets the build read the source that `gcloud builds submit` uploads.
for role in roles/artifactregistry.writer roles/run.developer roles/logging.logWriter \
  roles/storage.objectViewer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --condition=None --quiet \
    --member="serviceAccount:$BUILD_SA" --role="$role" >/dev/null
done
gcloud iam service-accounts add-iam-policy-binding "$JOB_SA" \
  --member="serviceAccount:$BUILD_SA" --role=roles/iam.serviceAccountUser >/dev/null

echo "==> Secret Manager: groq-api-key"
if ! gcloud secrets describe groq-api-key >/dev/null 2>&1; then
  if [[ -z "${GROQ_API_KEY:-}" ]]; then
    echo "GROQ_API_KEY is not set in this shell. Export it and re-run." >&2
    exit 1
  fi
  printf '%s' "$GROQ_API_KEY" |
    gcloud secrets create groq-api-key --data-file=- --replication-policy=automatic
fi
gcloud secrets add-iam-policy-binding groq-api-key \
  --member="serviceAccount:$JOB_SA" --role=roles/secretmanager.secretAccessor >/dev/null

echo "Done. Next: gcloud builds submit --config deploy/cloudbuild.yaml ."
