terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
  }

  # Keep state in a bucket so a team (and CI) share one source of truth:
  # backend "gcs" {
  #   bucket = "YOUR_PROJECT_ID-terraform-state"
  #   prefix = "verbatim"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
