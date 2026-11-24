variable "project_id" {
  description = "The Google Cloud project to deploy into."
  type        = string
}

variable "region" {
  description = "Where data, jobs and the agent run. London keeps NHS feedback in the UK."
  type        = string
  default     = "europe-west2"
}

variable "model_location" {
  description = "Vertex AI endpoint for Gemini. 'global' has the best availability; a region pins processing."
  type        = string
  default     = "global"
}

variable "github_repository" {
  description = "owner/name of the GitHub repository allowed to deploy, through keyless Workload Identity."
  type        = string
}

variable "create_bigquery_tables" {
  description = "External tables read their schema from the Parquet files, so enable after the first build-data run."
  type        = bool
  default     = false
}
