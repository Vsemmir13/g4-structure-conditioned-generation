args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 5) {
  stop("Usage: Rscript metrics/pqsfinder_metrics.R input.tsv output.csv min_score strand overlapping")
}

input_path <- args[[1]]
output_path <- args[[2]]
min_score <- as.integer(args[[3]])
strand_value <- args[[4]]
overlapping <- as.logical(args[[5]])

if (!requireNamespace("pqsfinder", quietly = TRUE)) {
  stop("Missing R package 'pqsfinder'. Install with: BiocManager::install('pqsfinder')")
}
if (!requireNamespace("Biostrings", quietly = TRUE)) {
  stop("Missing R package 'Biostrings'. Install with: BiocManager::install('Biostrings')")
}

suppressPackageStartupMessages(library(pqsfinder))
suppressPackageStartupMessages(library(Biostrings))

records <- read.delim(input_path, stringsAsFactors = FALSE)

metric_rows <- vector("list", nrow(records))

for (i in seq_len(nrow(records))) {
  seq_string <- toupper(records$seq[[i]])
  seq_string <- gsub("[^ACGTN]", "N", seq_string)

  pqs <- tryCatch(
    {
      invisible(capture.output(
        invisible(capture.output(
          result <- suppressWarnings(pqsfinder(
            DNAString(seq_string),
            min_score = min_score,
            strand = strand_value,
            overlapping = overlapping
          )),
          type = "message"
        )),
        type = "output"
      ))
      result
    },
    error = function(e) NULL
  )

  if (is.null(pqs) || length(pqs) == 0) {
    metric_rows[[i]] <- data.frame(
      row_id = records$row_id[[i]],
      source = records$source[[i]],
      model = records$model[[i]],
      generation = records$generation[[i]],
      class_level = records$class_level[[i]],
      samples_path = records$samples_path[[i]],
      pqs_count = 0L,
      pqs_max_score = 0,
      pqs_mean_score = 0,
      pqs_total_score = 0,
      pqs_max_width = 0,
      pqs_mean_width = 0,
      stringsAsFactors = FALSE
    )
  } else {
    metadata <- as.data.frame(elementMetadata(pqs))
    scores <- if ("score" %in% names(metadata)) metadata$score else rep(0, length(pqs))
    widths <- width(pqs)
    metric_rows[[i]] <- data.frame(
      row_id = records$row_id[[i]],
      source = records$source[[i]],
      model = records$model[[i]],
      generation = records$generation[[i]],
      class_level = records$class_level[[i]],
      samples_path = records$samples_path[[i]],
      pqs_count = length(pqs),
      pqs_max_score = max(scores),
      pqs_mean_score = mean(scores),
      pqs_total_score = sum(scores),
      pqs_max_width = max(widths),
      pqs_mean_width = mean(widths),
      stringsAsFactors = FALSE
    )
  }
}

metrics <- do.call(rbind, metric_rows)
write.csv(metrics, output_path, row.names = FALSE, quote = TRUE)
