library(gglasso)
library(tidyverse)
library(stringr)
library(ExclusiveLasso)

setwd("/Volumes/GoogleDrive/Shared drives/STA 275 Final Project/R-script")
path <- '../Data'
csf <- read.csv(paste0(path, "/data_imputed/Mean-Mode/csf.csv"))
uds <- read.csv(paste0(path, "/data_imputed/Mean-Mode/uds.csv"))
mri_FVIS <- read.csv(paste0(path, "/data_imputed/Mean-Mode/mri.csv"))
uds_dic <- read.csv(paste0(path,"/data-dictionary/uds_feature_dictionary_cleaned.csv"))
mri_dic <- read.csv(paste0(path, "/data-dictionary/mri_feature_dictionary_cleaned.csv"))

#mri_FVIS <- read.csv(paste0(path, "/pre-processed/mri_sub_rf.csv"))
#only take the baseline visit in MRI
#k <- mri_FVIS %>%  group_by(NACCID, datetime) %>% tally()
#which(k$n != 1)

mri_dic$Category <- ifelse(mri_dic$Category == "", NA, mri_dic$Category) #change "" to NA
mri_dic <- mri_dic[-which(is.na(mri_dic$Category)),] #remove the rows that does not have category

csf_dic <- data.frame(VariableName = colnames(csf))
csf_dic$Category[which(str_detect(csf_dic$VariableName, "CSF"))] <- "CSF"

## setting up the group numbers
#dic_csfname <- data.frame(var = colnames(csf))
#dic_csfname$category[which(str_detect(colnames(csf), "CSF"))] <- "CSF"
#hold <- unique(dic_csfname$category)
#dic_index <- data.frame(category = c(unique(uds_dic$Category), unique(mri_dic$Category), hold[!is.na(hold)])) #assign the group number
#dic_index$cat_index <- 1:dim(dic_index)[1]
# CSF variale dictionary
#dic_csfname$cat_index <- ifelse(dic_csfname$category == "CSF", 11, NA)

assign_dic <- function(dat, dat_dic){
  dic_dat <- data.frame(VariableName = colnames(dat))
  for (i in 1:length(colnames(dat))){
    dic_name <- dat_dic[which(dat_dic$VariableName == colnames(dat)[i]), "Category"]
    dic_name <- ifelse(identical(dic_name, character(0)), NA, dic_name)
    dic_dat$Category[i] <- dic_name
  }
  dic_index <- data.frame(category = unique(dic_dat$Category[!is.na(dic_dat$Category)])) #assign the group number
  dic_index$cat_index <- 1:dim(dic_index)[1]
  for (i in 1:length(colnames(dat))){
    index <- dic_index[which(dic_index$category == dic_dat$Category[i]), "cat_index"]
    index <- ifelse(identical(index, character(0)), NA, index)
    dic_dat$cat_index[i] <- index
  }
  out <- list(dic_dat, dic_index)
}


uds_dic[uds_dic$VariableName == "SEX", "Category"] = "SEX"
uds_dic[uds_dic$VariableName == "EDUC", "Category"] = "EDUC"
uds_dic[uds_dic$VariableName == "NACCAGE", "Category"] = "AGE"

# First, lasso only on UDS ------------------------------------------------------

# UDS variable dictionary
dic_udsname <- assign_dic(uds, uds_dic)[[1]]
dic_index <- assign_dic(uds, uds_dic)[[2]]
# Only use NACCGDS in GDS category (the rest are discarded)
rem_GDS <- grep("GDS",dic_udsname$Category)[grep("GDS",dic_udsname$Category) != grep("NACCGDS", dic_udsname$VariableName)]
uds_lasso <- uds[, -rem_GDS]
dic_udsname <- dic_udsname[-rem_GDS, ]

rem_demo <- which(dic_udsname$VariableName %in% c("NACCID","NACCADC","NACCVNUM","datetime","NACCALZP", "NACCUDSD"))
uds_lasso <- uds_lasso[, -rem_demo]
dic_udsname <- dic_udsname[-rem_demo, ]

## NACCAD5 -----------------------------------------------------------------
uds_lasso5 <- uds_lasso %>%  select(-NACCAD3)
uds_lasso5[which(uds_lasso5 == "", arr.ind = TRUE)] <- NA
uds_lasso5 <- uds_lasso5[complete.cases(uds_lasso5),]
uds_lasso5 <- uds_lasso5 %>% mutate(y5_fac = case_when(NACCAD5 == "Healthy" ~ 1,
                                                       NACCAD5 == "MCI-NonAD" ~ 2,
                                                       NACCAD5 == "Dementia-NonAD" ~ 3,
                                                       NACCAD5 == "MCI-AD" ~ 4,
                                                       NACCAD5 == "Dementia-AD" ~ 5))
y5 <- uds_lasso5$y5_fac
rem_y <- which(colnames(uds_lasso5) %in% c("NACCAD5", "y5_fac"))
uds_lasso5 <- uds_lasso5[, -rem_y]
rem_y <- which(dic_udsname$VariableName %in% c("NACCAD3", "NACCAD5", "y5_fac"))
dic_udsname <- dic_udsname[-rem_y, ]

orderi <- order(dic_udsname$cat_index)
dic_udsname <-  dic_udsname[orderi,]
uds_lasso5 <- uds_lasso5[, orderi]

identical(colnames(uds_lasso5), dic_udsname$VariableName)
uds_lasso5_mat <- as.matrix(uds_lasso5)


## NACCAD3 -----------------------------------------------------------------
dic_udsname <- assign_dic(uds, uds_dic)[[1]]
dic_index <- assign_dic(uds, uds_dic)[[2]]
# Only use NACCGDS in GDS category (the rest are discarded)
rem_GDS <- grep("GDS",dic_udsname$Category)[grep("GDS",dic_udsname$Category) != grep("NACCGDS", dic_udsname$VariableName)]
uds_lasso <- uds[, -rem_GDS]
dic_udsname <- dic_udsname[-rem_GDS, ]

rem_demo <- which(dic_udsname$VariableName %in% c("NACCID","NACCADC","NACCVNUM","datetime","NACCALZP", "NACCUDSD"))
uds_lasso <- uds_lasso[, -rem_demo]
dic_udsname <- dic_udsname[-rem_demo, ]

uds_lasso3 <- uds_lasso %>%  select(-NACCAD5)
uds_lasso3[which(uds_lasso3 == "", arr.ind = TRUE)] <- NA
uds_lasso3 <- uds_lasso3[complete.cases(uds_lasso3),]
uds_lasso3 <- uds_lasso3 %>% mutate(y3_fac = case_when(NACCAD3 == "Healthy" ~ 1,
                                                       NACCAD3 == "MCI-AD" ~ 2,
                                                       NACCAD3 == "Dementia-AD" ~ 3))
y3 <- uds_lasso3$y3_fac
rem_y <- which(colnames(uds_lasso3) %in% c("NACCAD3", "y3_fac"))
uds_lasso3 <- uds_lasso3[, -rem_y]
rem_y <- which(dic_udsname$VariableName %in% c("NACCAD3", "NACCAD5", "y3_fac"))
dic_udsname <- dic_udsname[-rem_y, ]

orderi <- order(dic_udsname$cat_index)
dic_udsname <-  dic_udsname[orderi,]
uds_lasso3 <- uds_lasso3[, orderi]
identical(colnames(uds_lasso3), dic_udsname$VariableName)
uds_lasso3_mat <- as.matrix(uds_lasso3)

### save files in RDS
run_uds <- list(list(uds_lasso5_mat, y5, dic_udsname), 
                   list(uds_lasso3_mat, y3, dic_udsname))
saveRDS(run_uds, file = "uds_lasso/run_uds.RDS")





# second: UDS and MRI -----------------------------------------------------
uds_mriFVIS <- merge(mri_FVIS, uds, by = "NACCID")
# MRI variable dictionary
dic_mriname <- assign_dic(mri_FVIS, mri_dic)[[1]]
# UDS variable dictionary
dic_udsname <- assign_dic(uds, uds_dic)[[1]]
# MRI and UDS variable dictionary
dic_uds_mriFVIS <- rbind(dic_mriname, dic_udsname)
dic_uds_mriFVIS <- assign_dic(uds_mriFVIS, dic_uds_mriFVIS)[[1]]

# Only use NACCGDS in GDS category (the rest are discarded)
rem_GDS <- grep("GDS",dic_uds_mriFVIS$Category)[grep("GDS",dic_uds_mriFVIS$Category) != grep("NACCGDS", dic_uds_mriFVIS$VariableName)]
uds_mriFVIS_lasso <- uds_mriFVIS[, -rem_GDS]
dic_uds_mriFVIS <- dic_uds_mriFVIS[-rem_GDS, ]
# remove "NACCID","NACCADC","NACCVNUM","datetime","NACCALZP" in the dataset
rem_demo <- which(colnames(uds_mriFVIS_lasso) %in% c("NACCID","NACCADC","NACCVNUM","datetime","NACCALZP",
                                                     "NACCVNUM.x", "NACCVNUM.y", "datetime.x", "datetime.y", 
                                                     "NACCUDSD", "datetime_UDS", "timediff", "within.a.year"))
uds_mriFVIS_lasso <- uds_mriFVIS_lasso[, -rem_demo]
rem_demo <- which(dic_uds_mriFVIS$VariableName %in% c("NACCID","NACCADC","NACCVNUM","datetime","NACCALZP",
                                                      "NACCVNUM.x", "NACCVNUM.y", "datetime.x", "datetime.y", 
                                                      "NACCUDSD", "datetime_UDS", "timediff", "within.a.year"))
dic_uds_mriFVIS <- dic_uds_mriFVIS[-rem_demo, ]

## NACCAD5 -----------------------------------------------------------------
dic_uds_mriFVIS5 <- dic_uds_mriFVIS
uds_mriFVIS_lasso5 <- uds_mriFVIS_lasso %>%  select(-NACCAD3)
uds_mriFVIS_lasso5[which(uds_mriFVIS_lasso5 == "", arr.ind = TRUE)] <- NA
uds_mriFVIS_lasso5 <- uds_mriFVIS_lasso5[complete.cases(uds_mriFVIS_lasso5),]
uds_mriFVIS_lasso5 <- uds_mriFVIS_lasso5 %>% mutate(y5_fac = case_when(NACCAD5 == "Healthy" ~ 1,
                                                                       NACCAD5 == "MCI-NonAD" ~ 2,
                                                                       NACCAD5 == "Dementia-NonAD" ~ 3,
                                                                       NACCAD5 == "MCI-AD" ~ 4,
                                                                       NACCAD5 == "Dementia-AD" ~ 5))
y5 <- uds_mriFVIS_lasso5$y5_fac
rem_y <- which(colnames(uds_mriFVIS_lasso5) %in% c("NACCAD5", "y5_fac"))
uds_mriFVIS_lasso5 <- uds_mriFVIS_lasso5[, -rem_y]
rem_y <- which(dic_uds_mriFVIS5$VariableName %in% c("NACCAD3", "NACCAD5", "y5_fac"))
dic_uds_mriFVIS5 <- dic_uds_mriFVIS5[-rem_y, ]

#reorder the group number
orderi <- order(dic_uds_mriFVIS5$cat_index)
dic_uds_mriFVIS5 <-  dic_uds_mriFVIS5[orderi,]
uds_mriFVIS_lasso5 <- uds_mriFVIS_lasso5[, orderi]

identical(colnames(uds_mriFVIS_lasso5), dic_uds_mriFVIS5$VariableName)
length(y5) - dim(uds_mriFVIS_lasso5)[1]
uds_mriFVIS_lasso5_mat <- as.matrix(uds_mriFVIS_lasso5)



## NACCAD3 -----------------------------------------------------------------
dic_uds_mriFVIS3 <- dic_uds_mriFVIS
uds_mriFVIS_lasso3 <- uds_mriFVIS_lasso %>%  select(-NACCAD5)
uds_mriFVIS_lasso3[which(uds_mriFVIS_lasso3 == "", arr.ind = TRUE)] <- NA
uds_mriFVIS_lasso3 <- uds_mriFVIS_lasso3[complete.cases(uds_mriFVIS_lasso3),]
uds_mriFVIS_lasso3 <- uds_mriFVIS_lasso3 %>% mutate(y3_fac = case_when(NACCAD3 == "Healthy" ~ 1,
                                                                       NACCAD3 == "MCI-NonAD" ~ 2,
                                                                       NACCAD3 == "Dementia-NonAD" ~ 3,
                                                                       NACCAD3 == "MCI-AD" ~ 4,
                                                                       NACCAD3 == "Dementia-AD" ~ 5))
y3 <- uds_mriFVIS_lasso3$y3_fac
rem_y <- which(colnames(uds_mriFVIS_lasso3) %in% c("NACCAD3", "y3_fac"))
uds_mriFVIS_lasso3 <- uds_mriFVIS_lasso3[, -rem_y]
rem_y <- which(dic_uds_mriFVIS3$VariableName %in% c("NACCAD3", "NACCAD5", "y3_fac"))
dic_uds_mriFVIS3 <- dic_uds_mriFVIS3[-rem_y, ]

#reorder the group number
orderi <- order(dic_uds_mriFVIS3$cat_index)
dic_uds_mriFVIS3 <-  dic_uds_mriFVIS3[orderi,]
uds_mriFVIS_lasso3 <- uds_mriFVIS_lasso3[, orderi]

identical(colnames(uds_mriFVIS_lasso3), dic_uds_mriFVIS3$VariableName)
uds_mriFVIS_lasso3_mat <- as.matrix(uds_mriFVIS_lasso3)

### save files in RDS
run_uds_mri <- list(list(uds_mriFVIS_lasso5_mat, y5, dic_uds_mriFVIS5), 
                   list(uds_mriFVIS_lasso3_mat, y3, dic_uds_mriFVIS5))
saveRDS(run_uds_mri, file = "uds_mri_lasso/run_uds_mri.RDS")


# Third: UDS, MRI, CSF -----------------------------------------------------
dat_all <- merge(mri_FVIS, uds, by = "NACCID")
dat_all <- merge(dat_all, csf, by = "NACCID")
# MRI, UDS, and CSF variable dictionary
dic_all <- data.frame(VariableName = c(mri_dic$VariableName, uds_dic$VariableName, csf_dic$VariableName),
                      Category = c(mri_dic$Category, uds_dic$Category, csf_dic$Category))
dic_all <- assign_dic(dat_all, dic_all)[[1]]
# Only use NACCGDS in GDS category (the rest are discarded)
rem_GDS <- grep("GDS",dic_all$Category)[grep("GDS",dic_all$Category) != grep("NACCGDS", dic_all$VariableName)]
dat_all_lasso <- dat_all[, -rem_GDS]
dic_all <- dic_all[-rem_GDS, ]
# remove "NACCID","NACCADC","NACCVNUM","datetime","NACCALZP" in the dataset
rem_demo <- which(dic_all$VariableName %in% c("NACCID","NACCADC","NACCVNUM","datetime","NACCALZP",
                                              "NACCVNUM.x", "NACCVNUM.y", "datetime.x", "datetime.y",
                                              "NACCADC.x", "NACCADC.y", "datetime_UDS",  "timediff",
                                              "within.a.year", "NACCUDSD"))
dat_all_lasso <- dat_all_lasso[, -rem_demo]
dic_all <- dic_all[-rem_demo, ]


## NACCAD5 -----------------------------------------------------------------
dic_all5 <- dic_all
dat_all_lasso5 <- dat_all_lasso %>%  select(-NACCAD3)
dat_all_lasso5[which(dat_all_lasso5 == "", arr.ind = TRUE)] <- NA
dat_all_lasso5 <- dat_all_lasso5[complete.cases(dat_all_lasso5),]
dat_all_lasso5 <- dat_all_lasso5 %>% mutate(y5_fac = case_when(NACCAD5 == "Healthy" ~ 1,
                                                                       NACCAD5 == "MCI-NonAD" ~ 2,
                                                                       NACCAD5 == "Dementia-NonAD" ~ 3,
                                                                       NACCAD5 == "MCI-AD" ~ 4,
                                                                       NACCAD5 == "Dementia-AD" ~ 5))
y5 <- dat_all_lasso5$y5_fac
rem_y <- which(colnames(dat_all_lasso5) %in% c("NACCAD5", "y5_fac"))
dat_all_lasso5 <- dat_all_lasso5[, -rem_y]
rem_y <- which(dic_all5$VariableName %in% c("NACCAD3", "NACCAD5", "y5_fac"))
dic_all5 <- dic_all5[-rem_y, ]

#reorder the group number
orderi <- order(dic_all5$cat_index)
dic_all5 <-  dic_all5[orderi,]
dat_all_lasso5 <- dat_all_lasso5[, orderi]

identical(colnames(dat_all_lasso5), dic_all5$VariableName)
dat_all_lasso5_mat <- as.matrix(dat_all_lasso5)


## NACCAD3 -----------------------------------------------------------------
dic_all3 <- dic_all
dat_all_lasso3 <- dat_all_lasso %>%  select(-NACCAD5)
dat_all_lasso3[which(dat_all_lasso3 == "", arr.ind = TRUE)] <- NA
dat_all_lasso3 <- dat_all_lasso3[complete.cases(dat_all_lasso3),]
dat_all_lasso3 <- dat_all_lasso3 %>% mutate(y3_fac = case_when(NACCAD3 == "Healthy" ~ 1,
                                                               NACCAD3 == "MCI-NonAD" ~ 2,
                                                               NACCAD3 == "Dementia-NonAD" ~ 3,
                                                               NACCAD3 == "MCI-AD" ~ 4,
                                                               NACCAD3 == "Dementia-AD" ~ 5))
y3 <- dat_all_lasso3$y3_fac
rem_y <- which(colnames(dat_all_lasso3) %in% c("NACCAD3", "y3_fac"))
dat_all_lasso3 <- dat_all_lasso3[, -rem_y]
rem_y <- which(dic_all3$VariableName %in% c("NACCAD3", "NACCAD5", "y3_fac"))
dic_all3 <- dic_all3[-rem_y, ]

#reorder the group number
orderi <- order(dic_all3$cat_index)
dic_all3 <-  dic_all3[orderi,]
dat_all_lasso3 <- dat_all_lasso3[, orderi]

identical(colnames(dat_all_lasso3), dic_all3$VariableName)
dat_all_lasso3_mat <- as.matrix(dat_all_lasso3)


### save files in RDS
run_all <- list(list(dat_all_lasso5_mat, y5, dic_all5), 
                       list(dat_all_lasso3_mat, y3, dic_all3))
saveRDS(run_all, file = "all_lasso/run_all.RDS")



# https://www.r-bloggers.com/2021/10/exclusive-lasso-and-group-lasso-using-r-code/
# couple questions:
# In CSF, what is NACCADC
# the NACCVNUM variable in mri dataset is not correctly specified. Several subjects have the several obs on the same visit day
# missing values in x not allowed, but if remove all missing values, nothing left in the data.
# exclusive lasso package cannot be download:
#library(devtools)
# install_github("DataSlingers/ExclusiveLasso")

#In CSF, data cleaning needed. variables like CSFTTMD and CSFTTMDX. One is completely empty


































