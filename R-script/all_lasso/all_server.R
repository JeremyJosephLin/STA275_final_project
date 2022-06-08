library(gglasso)
library(tidyverse)
library(stringr)
library(ExclusiveLasso)


# uds, mri, csf -----------------------------------------------------------
run_all <- readRDS("run_all.RDS")
## NACCAD5
dat_all_lasso5_mat <- run_all[[1]][[1]]
y5 <- run_all[[1]][[2]]
dic_all5 <- run_all[[1]][[3]]
identical(colnames(dat_all_lasso5_mat), dic_all5$VariableName)
dat_all_lasso5_mat <- as.matrix(dat_all_lasso5_mat)

#group lasso with selected lambda from GCV
gr_cv5 <- cv.gglasso(dat_all_lasso5_mat, y5, group = dic_all5$cat_index, 
                     loss="ls", pred.loss="L2")
gr5 <- gglasso(dat_all_lasso5_mat, y5, group = dic_all5$cat_index, lambda = c(gr_cv5$lambda.min, gr_cv5$lambda.1se))
#group lasso with selected lambda from GCV
ex_cv5 <- cv.exclusive_lasso(dat_all_lasso5_mat, y5, groups = dic_all5$cat_index)
ex5 <- exclusive_lasso(dat_all_lasso5_mat, y5, groups = dic_all5$cat_index, lambda= c(ex_cv5$lambda.min, ex_cv5$lambda.1se))


df.all5 <- data.frame(
  variable = dic_all5$VariableName,
  group = dic_all5$Category, 
  group_index = dic_all5$cat_index,
  Glasso_1se  = gr5$beta[,1],
  Glasso_min  = gr5$beta[,2],
  Exlasso_1se = ex5$coef[,1],
  Exlasso_min = ex5$coef[,2]
)

selected_lambda5 <- data.frame(glasso = gr5$lambda, exlasso = ex5$lambda)

## NACCAD3
dat_all_lasso3_mat <- run_all[[2]][[1]]
y3 <- run_all[[2]][[2]]
dic_all3 <- run_all[[2]][[3]]
identical(colnames(dat_all_lasso3_mat), dic_all3$VariableName)
dat_all_lasso3_mat <- as.matrix(dat_all_lasso3_mat)


#group lasso with selected lambda from GCV
gr_cv3 <- cv.gglasso(dat_all_lasso3_mat, y3, group = dic_all3$cat_index, 
                     loss="ls", pred.loss="L2")
gr3 <- gglasso(dat_all_lasso3_mat, y3, group = dic_all3$cat_index, lambda = c(gr_cv5$lambda.min, gr_cv5$lambda.1se))
#group lasso with selected lambda from GCV
ex_cv3 <- cv.exclusive_lasso(dat_all_lasso3_mat, y3, groups = dic_all3$cat_index)
ex3 <- exclusive_lasso(dat_all_lasso3_mat, y3, groups = dic_all3$cat_index, lambda= c(ex_cv5$lambda.min, ex_cv5$lambda.1se))


df.all3 <- data.frame(
  variable = dic_all3$VariableName,
  group = dic_all3$Category, 
  group_index = dic_all3$cat_index,
  Glasso_1se  = gr3$beta[,1],
  Glasso_min  = gr3$beta[,2],
  Exlasso_1se = ex3$coef[,1],
  Exlasso_min = ex3$coef[,2]
)

selected_lambda3 <- data.frame(glasso = gr3$lambda, exlasso = ex3$lambda)


result3 <- list(df.all5, selected_lambda5, df.all3, selected_lambda3)
saveRDS(result3, file = "result3.RDS")

pdf("cv_versus_lambda.pdf")
plot(gr_cv5, main = "group lasso with five levels of response variable")
plot(ex_cv5, main = "exclusive lasso with five levels of response variable")
plot(gr_cv3, main = "group lasso with three levels of response variable")
plot(ex_cv3, main = "exclusive lasso with three levels of response variable")
dev.off()




