library(ggplot2)
library(mgcv)
library(itsadug)
library(lme4)
library(lmerTest)
library(ggeffects)
library(MASS)
library(plyr)
library(dplyr)
library(tidyr)
library(tidyverse)

setwd('/Volumes/University/TiU/Research/ResearchTraineeship/2020_21-characters/LangCogn/')

names = read.csv('data/avgRatings_annotated.csv', header = T, sep = ',')
names = names %>%
  mutate(name = str_to_lower(name))


folder = 'output/predictions'
attributes = c('polarity', 'age', 'gender')
models = c('linear', 'nn') 
features = c('M5_lexical_True', 'M5_lexical_False', 'M2_lexical_False', 'M0_lexical_True', 'unigrams', 'phon_features')
rnd_ids = 0:49

preds = NULL
for (attribute in attributes) {
  for (model in models) {
    for (feature in features) {
      print(paste(attribute, model, feature, sep = ' - '))
      true_preds = paste(paste(folder, model, paste(attribute, 'predictions', feature, sep = '_'), sep='/'), 'csv', sep='.')
      df_true = read.csv(true_preds, header = T, sep = '\t')
      df_true = df_true %>%
        mutate(name = str_to_lower(name))
      df_true$features = feature
      df_true$attribute = attribute
      df_true$model = model
      df_true$run = 'true'
      df_true$error = df_true$true_rating - df_true$predicted_rating
      df_true$absolute_error = abs(df_true$error)
      preds = rbind(preds, df_true)
      
      for (i in rnd_ids) {
        rnd_preds = paste(paste(folder, model, paste(attribute, 'predictions', feature, paste('rnd', i, sep=''), sep = '_'), sep='/'), 'csv', sep='.')
        df_rnd = read.csv(rnd_preds, header = T, sep = '\t')
        df_rnd = df_rnd %>%
          mutate(name = str_to_lower(name))
        df_rnd$features = feature
        df_rnd$attribute = attribute
        df_rnd$model = model
        df_rnd$run = 'rnd'
        df_rnd$error = df_rnd$true_rating - df_rnd$predicted_rating
        df_rnd$absolute_error = abs(df_rnd$error)
        preds = rbind(preds, df_rnd)
      }
    }
  }
}

rm(df_rnd, df_true, attribute, feature, i, model, rnd_preds, true_preds)


df = merge(preds,
           dplyr::select(names, name, type, gender, age, polarity, rating.mean, attribute),
           by = c('name', 'attribute'), all.x = T)

rm(preds)

ggplot(data = df[df$attribute == 'age', ], aes(x = type, y = absolute_error)) +
  geom_boxplot(aes(color = run))+
  ggtitle('Absolute error, by feature, model, name type and run - Age') +
  facet_grid(model ~ features) +
  ylim(0, 1)

ggplot(data = df[df$attribute == 'polarity', ], aes(x = type, y = absolute_error)) +
  geom_boxplot(aes(color = run))+
  ggtitle('Absolute error, by feature, model, name type and run - Polarity') +
  facet_grid(model ~ features) +
  ylim(0, 1)

ggplot(data = df[df$attribute == 'gender', ], aes(x = type, y = absolute_error)) +
  geom_boxplot(aes(color = run))+
  ggtitle('Absolute error, by feature, model, name type and run - Gender') +
  facet_grid(model ~ features) +
  ylim(0, 1)

