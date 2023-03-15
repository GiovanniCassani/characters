#grid search analysis
library(tidyverse)
library(dplyr)

setwd('/Volumes/University/TiU/Research/ResearchTraineeship/2020_21-characters/LangCogn/')

############ gender
### neural networks
df_unigrams_gender = read.csv('output/gridSearch/nn/gender_gridSearch_unigrams.csv', sep = '\t', header = TRUE)
avg_df_unigrams_gender = df_unigrams_gender %>%
  filter(n_layers == 1) %>% 
  dplyr::group_by(nodes, dropout, act, lr, n_layers) %>% 
  dplyr::summarise(avg = mean(mse))
# units dropout     act     lr  layers    avg_mse
#    16    0.00    tanh  1e-04       1  0.4541999

df_phon_gender = read.csv('output/gridSearch/nn/gender_gridSearch_phon_features.csv', sep = '\t', header = TRUE)
avg_df_phon_gender = df_phon_gender %>%
  filter(n_layers == 1) %>% 
  dplyr::group_by(nodes, dropout, act, lr, n_layers) %>% 
  dplyr::summarise(avg = mean(mse))
# units dropout     act     lr  layers    avg_mse
#    16    0.00 sigmoid  1e-03       1  0.4868963

df_embeddings_gender = read.csv('output/gridSearch/nn/gender_gridSearch_M5_lexical_True.csv', sep = '\t', header = TRUE)
avg_df_embeddings_gender =  df_embeddings_gender %>%
  filter(n_layers == 1) %>% 
  dplyr::group_by(nodes, dropout, act, lr, n_layers) %>% 
  dplyr::summarise(avg = mean(mse))
# units dropout     act     lr  layers    avg_mse
#    50    0.00 sigmoid  1e-04       1  0.1919074





### linear models
df_unigrams_gender = read.csv('output/gridSearch/linear/gender_gridSearch_unigrams.csv', sep = '\t', header = TRUE)
avg_df_unigrams_gender = df_unigrams_gender %>%
  dplyr::group_by(alpha, l1_ratio) %>% 
  dplyr::summarise(avg = mean(mse))
# alpha l1_ratio    avg_mse
#  5.00     0.00  0.4648881  

df_phon_gender = read.csv('output/gridSearch/linear/gender_gridSearch_phon_features.csv', sep = '\t', header = TRUE)
avg_df_phon_gender = df_phon_gender %>%
  dplyr::group_by(alpha, l1_ratio) %>% 
  dplyr::summarise(avg = mean(mse))
# alpha l1_ratio    avg_mse
#  0.30     0.02  0.5088492  

df_embeddings_gender = read.csv('output/gridSearch/linear/gender_gridSearch_M5_lexical_True.csv', sep = '\t', header = TRUE)
avg_df_embeddings_gender =  df_embeddings_gender %>%
  dplyr::group_by(alpha, l1_ratio) %>% 
  dplyr::summarise(avg = mean(mse))
# alpha l1_ratio    avg_mse
#  5.00     0.00  0.2028326












############### age
### neural networks
df_unigrams_age = read.csv('output/gridSearch/nn/age_gridSearch_unigrams.csv', sep = '\t', header = TRUE)
avg_df_unigrams_age = df_unigrams_age %>%
  filter(n_layers == 1) %>% 
  dplyr::group_by(nodes, dropout, act, lr, n_layers) %>% 
  dplyr::summarise(avg = mean(mse))
# units dropout     act     lr  layers    avg_mse
#    16    0.00 sigmoid  1e-03       1  0.1511852

df_phon_age = read.csv('output/gridSearch/nn/age_gridSearch_phon_features.csv', sep = '\t', header = TRUE)
avg_df_phon_age = df_phon_age %>%
  filter(n_layers == 1) %>% 
  dplyr::group_by(nodes, dropout, act, lr, n_layers) %>% 
  dplyr::summarise(avg = mean(mse))
# units dropout     act     lr  layers    avg_mse
#    23    0.25 sigmoid  1e-03       1  0.1523232

df_embeddings_age = read.csv('output/gridSearch/nn/age_gridSearch_M5_lexical_True.csv', sep = '\t', header = TRUE)
avg_df_embeddings_age = df_embeddings_age %>%
  filter(n_layers == 1) %>% 
  dplyr::group_by(nodes, dropout, act, lr, n_layers) %>% 
  dplyr::summarise(avg = mean(mse))
# units dropout     act     lr  layers    avg_mse
#   150    0.25 sigmoid  1e-03       1  0.1056112 







### linear models
df_unigrams_age = read.csv('output/gridSearch/linear/age_gridSearch_unigrams.csv', sep = '\t', header = TRUE)
avg_df_unigrams_age = df_unigrams_age %>%
  dplyr::group_by(alpha, l1_ratio) %>% 
  dplyr::summarise(avg = mean(mse))
# alpha l1_ratio    avg_mse
#  0.05     0.08  0.1565639 

df_phon_age = read.csv('output/gridSearch/linear/age_gridSearch_phon_features.csv', sep = '\t', header = TRUE)
avg_df_phon_age = df_phon_age %>%
  dplyr::group_by(alpha, l1_ratio) %>% 
  dplyr::summarise(avg = mean(mse))
# alpha l1_ratio    avg_mse
#  0.15     0.24  0.1554118  

df_embeddings_age = read.csv('output/gridSearch/linear/age_gridSearch_M5_lexical_True.csv', sep = '\t', header = TRUE)
avg_df_embeddings_age =  df_embeddings_age %>%
  dplyr::group_by(alpha, l1_ratio) %>% 
  dplyr::summarise(avg = mean(mse))
# alpha l1_ratio    avg_mse
#  5.00     0.00  0.1185538











########## polarity
### neural networks
df_unigrams_polarity = read.csv('output/gridSearch/nn/polarity_gridSearch_unigrams.csv', sep = '\t', header = TRUE)
avg_df_unigrams_polarity = df_unigrams_polarity %>%
  filter(n_layers == 1) %>% 
  dplyr::group_by(nodes, dropout, act, lr, n_layers) %>% 
  dplyr::summarise(avg = mean(mse))
# units dropout     act     lr  layers    avg_mse
#    26    0.00    tanh  1e-04       1  0.1068567

df_phon_polarity = read.csv('output/gridSearch/nn/polarity_gridSearch_phon_features.csv', sep = '\t', header = TRUE)
avg_df_phon_polarity = df_phon_polarity %>%
  filter(n_layers == 1) %>% 
  dplyr::group_by(nodes, dropout, act, lr, n_layers) %>% 
  dplyr::summarise(avg = mean(mse))
# units dropout     act     lr  layers    avg_mse
#     8    0.00    tanh  1e-03       1  0.1015214

df_embeddings_polarity = read.csv('output/gridSearch/nn/polarity_gridSearch_M5_lexical_True.csv', sep = '\t', header = TRUE)
avg_df_embeddings_polarity = df_embeddings_polarity %>%
  filter(n_layers == 1) %>% 
  dplyr::group_by(nodes, dropout, act, lr, n_layers) %>% 
  dplyr::summarise(avg = mean(mse))
# units dropout     act     lr  layers    avg_mse
#   600    0.00    relu  1e-04       1  0.1167228






### linear models
df_unigrams_polarity = read.csv('output/gridSearch/linear/polarity_gridSearch_unigrams.csv', sep = '\t', header = TRUE)
avg_df_unigrams_polarity = df_unigrams_polarity %>%
  dplyr::group_by(alpha, l1_ratio) %>% 
  dplyr::summarise(avg = mean(mse))
# alpha l1_ratio    avg_mse
#  0.20     0.02  0.1205185

df_phon_polarity = read.csv('output/gridSearch/linear/polarity_gridSearch_phon_features.csv', sep = '\t', header = TRUE)
avg_df_phon_polarity = df_phon_polarity %>%
  dplyr::group_by(alpha, l1_ratio) %>% 
  dplyr::summarise(avg = mean(mse))
# alpha l1_ratio    avg_mse
#  0.10     0.02  0.1098404

df_embeddings_polarity = read.csv('output/gridSearch/linear/polarity_gridSearch_M5_lexical_True.csv', sep = '\t', header = TRUE)
avg_df_embeddings_polarity =  df_embeddings_polarity %>%
  dplyr::group_by(alpha, l1_ratio) %>% 
  dplyr::summarise(avg = mean(mse))
# alpha l1_ratio    avg_mse
#  0.05     0.20  0.1283636

