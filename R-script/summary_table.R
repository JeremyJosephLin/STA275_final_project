
uds <- read.csv("~/Google Drive/Shared drives/STA 275 Final Project/Data/data_cleaned/uds.csv")
mri <- read.csv("~/Google Drive/Shared drives/STA 275 Final Project/Data/data_cleaned/mri.csv")
csf <- read.csv("~/Google Drive/Shared drives/STA 275 Final Project/Data/data_cleaned/csf.csv")

library(dplyr)
library(gtsummary)
library(kableExtra)

base_table <- uds %>% 
  dplyr::select(SEX, EDUC, NACCAGE, NACCAPOE, NACCAD3)


base_table$NACCAD3[base_table$NACCAD3 == ""] <- NA

base_table$SEX[base_table$SEX == 1] <- "Male"
base_table$SEX[base_table$SEX == 2] <- "Female"

base_table$NACCAPOE <- as.numeric(base_table$NACCAPOE)

base_table %>% 
  tbl_summary(
    by = NACCAD3,
    type = list(NACCAPOE ~ "continuous"),
    statistic = list(
      all_continuous() ~ "{mean} ({sd})"
    ))   %>% 
  modify_spanning_header(c("stat_1", "stat_2") ~ "**Cognitive State**") %>%
  as_kable_extra(
    format = "latex",
    digits = 4,
    booktabs = TRUE,
    linesep = "",
    align = c("l"),
      caption = "Baseline summary statistic stratified by AD"
  ) %>%
  kable_styling(latex_options = "HOLD_position",
                font_size = 6)