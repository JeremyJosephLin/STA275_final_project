library(gglasso)
library(tidyverse)
library(stringr)
library(ExclusiveLasso)

# uds ---------------------------------------------------------------------

run_uds <- readRDS("run_uds.RDS")

# NACCAD5
uds_lasso5_mat <- run_uds[[1]][[1]]
y5 <- run_uds[[1]][[2]]
dic_udsname <- run_uds[[1]][[3]]
identical(colnames(uds_lasso5_mat), dic_udsname$VariableName)
uds_lasso5_mat <- as.matrix(uds_lasso5_mat)

#group lasso with selected lambda from GCV
gr_cv5 <- cv.gglasso(uds_lasso5_mat, y5, group = dic_udsname$cat_index, 
                     loss="ls", pred.loss="L2", lambda = seq(0.1,2, length = 40))
gr5 <- gglasso(uds_lasso5_mat, y5, group = dic_udsname$cat_index, lambda = c(gr_cv5$lambda.min, gr_cv5$lambda.1se))
print("finish gr5")
#group lasso with selected lambda from GCV
ex_cv5 <- cv.exclusive_lasso(uds_lasso5_mat, y5, groups = dic_udsname$cat_index, lambda = seq(0.1,2, length = 40))
ex5 <- exclusive_lasso(uds_lasso5_mat, y5, groups = dic_udsname$cat_index, lambda= c(ex_cv5$lambda.min, ex_cv5$lambda.1se))
print("finish ex5")

df.comp_uds5 <- data.frame(
  variable = dic_udsname$VariableName,
  group = dic_udsname$Category, 
  group_index = dic_udsname$cat_index,
  Glasso_1se  = gr5$beta[,1],
  Glasso_min  = gr5$beta[,2],
  Exlasso_1se = ex5$coef[,1],
  Exlasso_min = ex5$coef[,2]
)

selected_lambda5 <- data.frame(glasso = gr5$lambda, exlasso = ex5$lambda)

#group_lambda5 <- list(paste0("The optimal value of lambda that gives minimum cross validation error cvm is", gr_cv$lambda.min), 
#                         paste0("The largest value of lambda such that error is within 1 standard error of the minimum is ", gr_cv$lambda.1se))



# NACCAD3
uds_lasso3_mat <- run_uds[[2]][[1]]
y3 <- run_uds[[2]][[2]]
dic_udsname <- run_uds[[2]][[3]]
identical(colnames(uds_lasso3_mat), dic_udsname$VariableName)
uds_lasso3_mat <- as.matrix(uds_lasso3_mat)

#group lasso with selected lambda from GCV
gr_cv3 <- cv.gglasso(uds_lasso3_mat, y3, group = dic_udsname$cat_index, 
                     loss="ls", pred.loss="L2",  lambda = seq(0.1,2, length = 40))
gr3 <- gglasso(uds_lasso3_mat, y3, group = dic_udsname$cat_index, lambda = c(gr_cv3$lambda.min, gr_cv3$lambda.1se))
print("finish gr3")
#group lasso with selected lambda from GCV
ex_cv3 <- cv.exclusive_lasso(uds_lasso3_mat, y3, groups = dic_udsname$cat_index,  lambda = seq(0.1,2, length = 40))
ex3 <- exclusive_lasso(uds_lasso3_mat, y3, groups = dic_udsname$cat_index, lambda= c(ex_cv3$lambda.min, ex_cv3$lambda.1se))
print("finish ex3")

selected_lambda3 <- data.frame(glasso = gr3$lambda, exlasso = ex3$lambda)

df.comp_uds3 <- data.frame(
  variable = dic_udsname$VariableName,
  group = dic_udsname$Category, 
  group_index = dic_udsname$cat_index,
  Glasso_1se  = gr3$beta[,1],
  Glasso_min  = gr3$beta[,2],
  Exlasso_1se = ex3$coef[,1],
  Exlasso_min = ex3$coef[,2]
)


result1 <- list(df.comp_uds5, selected_lambda5, df.comp_uds3, selected_lambda3)
saveRDS(result1, file = "result1.RDS")

pdf("cv_versus_lambda.pdf")
plot(gr_cv5, main = "group lasso with five levels of response variable")
plot(ex_cv5, main = "exclusive lasso with five levels of response variable")
plot(gr_cv3, main = "group lasso with three levels of response variable")
plot(ex_cv3, main = "exclusive lasso with three levels of response variable")
dev.off()




























