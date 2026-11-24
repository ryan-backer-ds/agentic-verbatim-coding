# The batch job: one container image, one command per execution.
#
#   gcloud run jobs execute verbatim-batch --region=europe-west2 --args=build-data --wait
#   gcloud run jobs execute verbatim-batch --region=europe-west2 \
#     --args=code,--split,test --wait
#
# Terraform owns the job's shape; Cloud Build (deploy/cloudbuild.yaml) rolls out
# new images, so the image field is ignored here after the first apply.

resource "google_cloud_run_v2_job" "batch" {
  name                = "verbatim-batch"
  location            = var.region
  deletion_protection = false

  template {
    task_count = 1

    template {
      service_account       = google_service_account.batch.email
      max_retries           = 1
      timeout               = "3600s"
      execution_environment = "EXECUTION_ENVIRONMENT_GEN2" # needed for Cloud Storage volumes

      containers {
        image = "${local.image}:latest"
        args  = ["check-models"]

        resources {
          limits = {
            cpu    = "1"
            memory = "2Gi"
          }
        }

        # Gemini through Vertex AI as the job's own identity: no API key exists.
        env {
          name  = "LLM__PROVIDER"
          value = "vertex"
        }
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        env {
          name  = "GOOGLE_CLOUD_LOCATION"
          value = var.model_location
        }
        env {
          name  = "CODING__CONCURRENCY"
          value = "16"
        }
        env {
          name  = "SAFETY__TRACING"
          value = "cloud_trace"
        }

        volume_mounts {
          name       = "data"
          mount_path = "/data"
        }
      }

      volumes {
        name = "data"
        gcs {
          bucket        = google_storage_bucket.data.name
          read_only     = false
          mount_options = ["uid=1000", "gid=1000"] # the image runs as user 1000
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [template[0].template[0].containers[0].image]
  }
  depends_on = [google_project_service.apis, google_storage_bucket_iam_member.batch_data]
}
