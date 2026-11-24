# Values the deploy workflow and scripts need. Copy them into the GitHub
# repository's Actions variables (Settings > Secrets and variables > Actions).

output "image" {
  value = local.image
}

output "data_bucket" {
  value = google_storage_bucket.data.name
}

output "workload_identity_provider" {
  description = "GCP_WORKLOAD_IDENTITY_PROVIDER for deploy.yml"
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deployer_service_account" {
  description = "GCP_DEPLOYER_SERVICE_ACCOUNT for deploy.yml"
  value       = google_service_account.deployer.email
}

output "model_armor_template" {
  description = "SAFETY__MODEL_ARMOR_TEMPLATE"
  value       = google_model_armor_template.prompts.id
}
