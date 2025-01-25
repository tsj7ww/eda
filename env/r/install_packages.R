# Install required system dependencies first
system_deps <- function() {
  system('apt-get update && apt-get install -y libxml2-dev libssl-dev libcurl4-openssl-dev')
}

# Function to handle package installation with error handling
install_packages_safely <- function(packages) {
  # Set CRAN mirror
  options(repos = c(CRAN = "https://cran.rstudio.com/"))
  
  # Read the package list
  source("/tmp/packages.R")
  
  # Function to safely install a single package
  install_single_package <- function(pkg) {
    tryCatch({
      if (!require(pkg, character.only = TRUE)) {
        install.packages(pkg, dependencies = TRUE)
      }
      TRUE
    }, error = function(e) {
      message(sprintf("Failed to install %s: %s", pkg, e$message))
      FALSE
    })
  }
  
  # Install packages
  results <- sapply(packages, install_single_package)
  
  # Report results
  successful <- sum(results)
  failed <- sum(!results)
  
  message(sprintf("\nInstallation complete:\n- %d packages installed successfully\n- %d packages failed\n",
                 successful, failed))
  
  # Return names of failed packages
  if (failed > 0) {
    failed_pkgs <- names(results)[!results]
    message("Failed packages: ", paste(failed_pkgs, collapse = ", "))
  }
}

# Install packages
system_deps()
install_packages_safely(packages)