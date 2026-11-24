# Verbatim on Google Cloud: storage, analytics, safety and the container registry.
# Service accounts and permissions are in iam.tf, the batch job in run.tf, and
# the keyless GitHub deploy identity in github.tf.

locals {
  apis = [
    "aiplatform.googleapis.com",        # Gemini on Vertex AI, Agent Runtime
    "artifactregistry.googleapis.com",  # container images
    "bigquery.googleapis.com",          # analytics over coded answers
    "cloudbuild.googleapis.com",        # image builds
    "cloudtrace.googleapis.com",        # agent traces
    "iamcredentials.googleapis.com",    # Workload Identity Federation
    "modelarmor.googleapis.com",        # prompt-injection screening
    "run.googleapis.com",               # the batch job
    "sts.googleapis.com",               # Workload Identity Federation
  ]
  image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/verbatim"
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  repository_id = "verbatim"
  location      = var.region
  format        = "DOCKER"
  description   = "Verbatim batch and agent images"

  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }
  depends_on = [google_project_service.apis]
}

# Raw, processed and coded data. The batch job mounts it at /data.
resource "google_storage_bucket" "data" {
  name                        = "${var.project_id}-verbatim-data"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true # a bad run can be rolled back
  }
}

# Where Agent Runtime stages the packaged agent.
resource "google_storage_bucket" "agent_staging" {
  name                        = "${var.project_id}-verbatim-agent-staging"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# Analysts query the coded answers in BigQuery, straight from the Parquet files.
resource "google_bigquery_dataset" "verbatim" {
  dataset_id  = "verbatim"
  location    = var.region
  description = "Coded NHS Friends and Family Test answers"
  depends_on  = [google_project_service.apis]
}

resource "google_bigquery_table" "responses" {
  count               = var.create_bigquery_tables ? 1 : 0
  dataset_id          = google_bigquery_dataset.verbatim.dataset_id
  table_id            = "responses"
  deletion_protection = false

  external_data_configuration {
    source_format = "PARQUET"
    autodetect    = true
    source_uris   = ["gs://${google_storage_bucket.data.name}/processed/responses.parquet"]
  }
}

resource "google_bigquery_table" "coded_answers" {
  count               = var.create_bigquery_tables ? 1 : 0
  dataset_id          = google_bigquery_dataset.verbatim.dataset_id
  table_id            = "coded_answers"
  deletion_protection = false

  external_data_configuration {
    source_format = "PARQUET"
    autodetect    = true
    source_uris   = ["gs://${google_storage_bucket.data.name}/coded/*.parquet"]
  }
}

# Screens analyst questions before they reach a model (SAFETY__MODEL_ARMOR_TEMPLATE).
resource "google_model_armor_template" "prompts" {
  location    = var.region
  template_id = "verbatim-prompts"

  filter_config {
    pi_and_jailbreak_filter_settings {
      filter_enforcement = "ENABLED"
      confidence_level   = "MEDIUM_AND_ABOVE"
    }
    malicious_uri_filter_settings {
      filter_enforcement = "ENABLED"
    }
    sdp_settings {
      basic_config {
        filter_enforcement = "ENABLED"
      }
    }
  }
  depends_on = [google_project_service.apis]
}
