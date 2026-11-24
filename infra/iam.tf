# One service account per workload, each with only the roles it uses.
# Nothing here creates a key: workloads use their attached identity, and GitHub
# Actions borrows the deployer identity through Workload Identity Federation.

resource "google_service_account" "batch" {
  account_id   = "verbatim-batch"
  display_name = "Verbatim batch job (data build, coding runs)"
}

resource "google_service_account" "agent" {
  account_id   = "verbatim-agent"
  display_name = "Verbatim insights agent on Agent Runtime"
}

resource "google_service_account" "deployer" {
  account_id   = "verbatim-deployer"
  display_name = "Verbatim deployments from GitHub Actions"
}

locals {
  project_roles = {
    batch = [
      "roles/aiplatform.user", # call Gemini and embeddings on Vertex AI
      "roles/cloudtrace.agent",
    ]
    agent = [
      "roles/aiplatform.user",
      "roles/modelarmor.user", # screen questions
      "roles/cloudtrace.agent",
    ]
    deployer = [
      "roles/artifactregistry.writer",
      "roles/cloudbuild.builds.editor",
      "roles/logging.logWriter",       # Cloud Build logs when building as this account
      "roles/run.developer",           # roll the batch job onto a new image
      "roles/aiplatform.user",         # create and update the agent runtime
      "roles/storage.objectAdmin",     # upload build sources and agent packages
      "roles/serviceusage.serviceUsageConsumer",
    ]
  }
  accounts = {
    batch    = google_service_account.batch.email
    agent    = google_service_account.agent.email
    deployer = google_service_account.deployer.email
  }
  project_bindings = merge([
    for account, roles in local.project_roles : {
      for role in roles : "${account}:${role}" => { email = local.accounts[account], role = role }
    }
  ]...)
}

resource "google_project_iam_member" "roles" {
  for_each = local.project_bindings
  project  = var.project_id
  role     = each.value.role
  member   = "serviceAccount:${each.value.email}"
}

# Data access is per bucket, not project-wide.
resource "google_storage_bucket_iam_member" "batch_data" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.batch.email}"
}

resource "google_storage_bucket_iam_member" "agent_data" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.agent.email}"
}

resource "google_bigquery_dataset_iam_member" "batch_writes_tables" {
  dataset_id = google_bigquery_dataset.verbatim.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.batch.email}"
}

# The deployer may run workloads as the batch and agent accounts, and builds as itself.
resource "google_service_account_iam_member" "deployer_acts_as" {
  for_each           = local.accounts
  service_account_id = "projects/${var.project_id}/serviceAccounts/${each.value}"
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}
