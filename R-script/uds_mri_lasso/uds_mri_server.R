library(gglasso)
library(tidyverse)
library(stringr)
library(ExclusiveLasso)


# uds and mri -------------------------------------------------------------
run_uds_mri <-readRDS("run_uds_mri.RDS")
## NACCAD5
uds_mriFVIS_lasso5_mat <- run_uds_mri[[1]][[1]]
y5 <- run_uds_mri[[1]][[2]]
dic_uds_mriFVIS5 <- run_uds_mri[[1]][[3]]
identical(colnames(uds_mriFVIS_lasso5_mat), dic_uds_mriFVIS5$VariableName)
uds_mriFVIS_lasso5_mat <- as.matrix(uds_mriFVIS_lasso5_mat)


#group lasso with selected lambda from GCV
gr_cv5 <- cv.gglasso(uds_mriFVIS_lasso5_mat, y5, group = dic_uds_mriFVIS5$cat_index,
                     loss="ls", pred.loss="L2")
gr5 <- gglasso(uds_mriFVIS_lasso5_mat, y5, group = dic_uds_mriFVIS5$cat_index, lambda = c(gr_cv5$lambda.min, gr_cv5$lambda.1se))
#group lasso with selected lambda from GCV
ex_cv5 <- cv.exclusive_lasso(uds_mriFVIS_lasso5_mat, y5, groups = dic_uds_mriFVIS5$cat_index)
ex5 <- exclusive_lasso(uds_mriFVIS_lasso5_mat, y5, groups = dic_uds_mriFVIS5$cat_index, lambda= c(ex_cv5$lambda.min, ex_cv5$lambda.1se))


df.comp_uds_mri5 <- data.frame(
  Variable = dic_uds_mriFVIS5$VariableName,
  group = dic_uds_mriFVIS5$Category, 
  group_index = dic_uds_mriFVIS5$cat_index,
  Glasso_1se  = gr5$beta[,1],
  Glasso_min  = gr5$beta[,2],
  Exlasso_1se = ex5$coef[,1],
  Exlasso_min = ex5$coef[,2]
)

selected_lambda5 <- data.frame(glasso = gr5$lambda, exlasso = ex5$lambda)


## NACCAD3
uds_mriFVIS_lasso3_mat <- run_uds_mri[[2]][[1]]
y3 <- run_uds_mri[[2]][[2]]
dic_uds_mriFVIS3 <- run_uds_mri[[2]][[3]]
identical(colnames(uds_mriFVIS_lasso3_mat), dic_uds_mriFVIS3$VariableName)

#group lasso with selected lambda from GCV
gr_cv3 <- cv.gglasso(uds_mriFVIS_lasso3_mat, y3, group = dic_uds_mriFVIS3$cat_index,
                     loss="ls", pred.loss="L2")
gr3 <- gglasso(uds_mriFVIS_lasso3_mat, y3, group = dic_uds_mriFVIS3$cat_index, lambda = c(gr_cv3$lambda.min, gr_cv3$lambda.1se))
#group lasso with selected lambda from GCV
ex_cv3 <- cv.exclusive_lasso(uds_mriFVIS_lasso3_mat, y3, group = dic_uds_mriFVIS3$cat_index)
ex3 <- exclusive_lasso(uds_mriFVIS_lasso3_mat, y3, group = dic_uds_mriFVIS3$cat_index, lambda= c(ex_cv3$lambda.min, ex_cv3$lambda.1se))


df.comp_uds_mri3 <- data.frame(
  Variable = dic_uds_mriFVIS3$VariableName,
  group = dic_uds_mriFVIS3$Category, 
  group_index = dic_uds_mriFVIS3$cat_index,
  Glasso_1se  = gr3$beta[,1],
  Glasso_min  = gr3$beta[,2],
  Exlasso_1se = ex3$coef[,1],
  Exlasso_min = ex3$coef[,2]
)

selected_lambda3 <- data.frame(glasso = gr3$lambda, exlasso = ex3$lambda)

result2 <- list(df.comp_uds_mri5, selected_lambda5, df.comp_uds_mri3, selected_lambda3)
saveRDS(result2, file = "result2.RDS")

pdf("cv_versus_lambda.pdf")
plot(gr_cv5, main = "group lasso with five levels of response variable")
plot(ex_cv5, main = "exclusive lasso with five levels of response variable")
plot(gr_cv3, main = "group lasso with three levels of response variable")
plot(ex_cv3, main = "exclusive lasso with three levels of response variable")
dev.off()
