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

load("output/MAE.RData")
rm(preds)

df.subset = df %>%
  mutate_if(is.character,as.factor) %>%
  filter(features %in% c("M5_lexical_False", "M2_lexical_False", "unigrams", "phon_features", "M0_lexical_True", "M5_lexical_True")) %>%
  filter(run != 'avg') %>%
  mutate(features=recode(
    features,
    "M0_lexical_True" = 'lexical', 
    "M2_lexical_False" = 'bigrams', 
    "M5_lexical_False" = 'ngrams', 
    "M5_lexical_True" = 'all',
    "phon_features" = 'phon_features', 
    "unigrams" = 'letters')) %>%
  mutate(run=recode(run,
                    "rnd" = 'random mappings', 
                    "true" = 'true mappings')) %>%
  mutate(model=recode(model,
                      "linear" = 'elastic net', 
                      "nn" = 'neural network')) %>%
  mutate(features = fct_relevel(
    features, 
    c('lexical', 'all', 'ngrams', 'bigrams', 'letters', 'phon_features'))
    )

df.subset = droplevels(df.subset)


##### NEURAL NETWORKS #####
##### age ####
df.age = df.subset[df.subset$attribute == 'age' & df.subset$model == 'neural network' & df.subset$run =='true mappings', ]

lm.age.base = lm(true_rating ~ type, data = df.age)
lm.age.preds = lm(true_rating ~ type + predicted_rating + features, data = df.age)
lm.age.preds.int = lm(true_rating ~ type*features*predicted_rating, data = df.age)

AIC(lm.age.base) # 701.2538
AIC(lm.age.preds) # 214.4254
AIC(lm.age.preds.int) # 223.1559  220.9543 for type*features, 221.9573 for features*rating

summary(lm.age.base)
summary(lm.age.preds)
summary(lm.age.preds.int)

age.preds = data.frame(
  ggpredict(lm.age.preds.int, terms = c(
    "predicted_rating[all]",
    "features[all]",
    "type[all]")
  )
)
names(age.preds)[names(age.preds) == 'x'] <- 'modelPrediction'
names(age.preds)[names(age.preds) == 'group'] <- 'FeatureSpace'
names(age.preds)[names(age.preds) == 'facet'] <- 'type'
names(age.preds)[names(age.preds) == 'predicted'] <- 'Rating'

pdf(file = "plots/predictions/lm.nn.age.pdf",  
    width = 7,
    height = 5)
ggplot(data = age.preds, aes(x = modelPrediction, y = Rating)) +
  geom_line(aes(color = FeatureSpace)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = FeatureSpace, fill=FeatureSpace), alpha = 0.4) +
  geom_point(aes(x=predicted_rating, y=true_rating, color=features), size=0.5, data=df.age) +
  facet_grid(. ~ type) +
  ylim(-1, 1) +
  ggtitle('Prediction relation with age rating, by features and name type') +
  scale_fill_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred",
    "letters" = "firebrick1",
    "phon_features" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred",
    "letters" = "firebrick1",
    "phon_features" = "deeppink1"
  )) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) 
dev.off()




##### gender ####
df.gen = df.subset[df.subset$attribute == 'gender' & df.subset$model == 'neural network' & df.subset$run =='true mappings', ]

lm.gen.base = lm(true_rating ~ type, data = df.gen)
lm.gen.preds = lm(true_rating ~ type + predicted_rating + features, data = df.gen)
lm.gen.preds.int = lm(true_rating ~ type + features*predicted_rating, data = df.gen)

AIC(lm.gen.base) # 2395.074
AIC(lm.gen.preds) # 1578.329
AIC(lm.gen.preds.int) # 1597.411  1591.039 for type*features, 1578.184 for features*predicted rating

summary(lm.gen.base)
summary(lm.gen.preds)
summary(lm.gen.preds.int)

gen.preds = data.frame(
  ggpredict(lm.gen.preds.int, terms = c(
    "predicted_rating[all]",
    "features[all]",
    "type[all]")
  )
)
names(gen.preds)[names(gen.preds) == 'x'] <- 'modelPrediction'
names(gen.preds)[names(gen.preds) == 'group'] <- 'FeatureSpace'
names(gen.preds)[names(gen.preds) == 'facet'] <- 'type'
names(gen.preds)[names(gen.preds) == 'predicted'] <- 'Rating'


pdf(file = "plots/predictions/lm.nn.gender.pdf",  
    width = 7,
    height = 5)
ggplot(data = gen.preds, aes(x = modelPrediction, y = Rating)) +
  geom_line(aes(color = FeatureSpace)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = FeatureSpace, fill=FeatureSpace), alpha = 0.4) +
  geom_point(aes(x=predicted_rating, y=true_rating, color=features), size=0.5, data=df.gen) +
  facet_grid(. ~ type) +
  ylim(-1, 1) +
  ggtitle('Prediction relation with gender rating, by features and name type') +
  scale_fill_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred",
    "letters" = "firebrick1",
    "phon_features" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred",
    "letters" = "firebrick1",
    "phon_features" = "deeppink1"
  )) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) 
dev.off()




##### polarity #####
df.pol = df.subset[df.subset$attribute == 'polarity' & df.subset$model == 'neural network' & df.subset$run =='true mappings', ]

lm.pol.base = lm(true_rating ~ type, data = df.pol)
lm.pol.preds = lm(true_rating ~ type + predicted_rating + features, data = df.pol)
lm.pol.preds.int = lm(true_rating ~ type*features*predicted_rating, data = df.pol)

AIC(lm.pol.base) # 268.8424
AIC(lm.pol.preds) # -21.06982
AIC(lm.pol.preds.int) # 7.055563

summary(lm.pol.base)
summary(lm.pol.preds)
summary(lm.pol.preds.int)

pol.preds = data.frame(
  ggpredict(lm.pol.preds.int, terms = c(
    "predicted_rating[all]",
    "features[all]",
    "type[all]")
  )
)
names(pol.preds)[names(pol.preds) == 'x'] <- 'modelPrediction'
names(pol.preds)[names(pol.preds) == 'group'] <- 'FeatureSpace'
names(pol.preds)[names(pol.preds) == 'facet'] <- 'type'
names(pol.preds)[names(pol.preds) == 'predicted'] <- 'Rating'


pdf(file = "plots/predictions/lm.nn.polarity.pdf",  
    width = 7,
    height = 5)
ggplot(data = pol.preds, aes(x = modelPrediction, y = Rating)) +
  geom_line(aes(color = FeatureSpace)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = FeatureSpace, fill=FeatureSpace), alpha = 0.4) +
  geom_point(aes(x=predicted_rating, y=true_rating, color=features), size=0.5, data=df.pol) +
  facet_grid(. ~ type) +
  ylim(-1, 1) +
  ggtitle('Prediction relation with polarity rating, by features and name type') +
  scale_fill_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred",
    "letters" = "firebrick1",
    "phon_features" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred",
    "letters" = "firebrick1",
    "phon_features" = "deeppink1"
  )) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) 
dev.off()



##### ELASTICNET ##### 
##### age ####
df.age = df.subset[df.subset$attribute == 'age' & df.subset$model == 'elastic net' & df.subset$run =='true mappings', ]

lm.en.age.base = lm(true_rating ~ type, data = df.age)
lm.en.age.preds = lm(true_rating ~ type + predicted_rating + features, data = df.age)
lm.en.age.preds.int = lm(true_rating ~ type*features*predicted_rating, data = df.age)

AIC(lm.en.age.base) # 701.2538
AIC(lm.en.age.preds) # 593.0371
AIC(lm.en.age.preds.int) # 574.8721

summary(lm.en.age.base)
summary(lm.en.age.preds)
summary(lm.en.age.preds.int)

age.preds.en = data.frame(
  ggpredict(lm.en.age.preds.int, terms = c(
    "predicted_rating[all]",
    "features[all]",
    "type[all]")
  )
)
names(age.preds.en)[names(age.preds.en) == 'x'] <- 'modelPrediction'
names(age.preds.en)[names(age.preds.en) == 'group'] <- 'FeatureSpace'
names(age.preds.en)[names(age.preds.en) == 'facet'] <- 'type'
names(age.preds.en)[names(age.preds.en) == 'predicted'] <- 'Rating'

pdf(file = "plots/predictions/lm.en.age.pdf",  
    width = 7,
    height = 5)
ggplot(data = age.preds.en, aes(x = modelPrediction, y = Rating)) +
  geom_line(aes(color = FeatureSpace)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = FeatureSpace, fill=FeatureSpace), alpha = 0.4) +
  geom_point(aes(x=predicted_rating, y=true_rating, color=features), size=0.5, data=df.age) +
  facet_grid(. ~ type) +
  ylim(-1, 1) +
  ggtitle('Prediction relation with age rating, by features and name type') +
  scale_fill_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred",
    "letters" = "firebrick1",
    "phon_features" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred",
    "letters" = "firebrick1",
    "phon_features" = "deeppink1"
  )) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) 
dev.off()




##### gender ####
df.gen = df.subset[df.subset$attribute == 'gender' & df.subset$model == 'elastic net' & df.subset$run =='true mappings', ]

lm.en.gen.base = lm(true_rating ~ type, data = df.gen)
lm.en.gen.preds = lm(true_rating ~ type + predicted_rating + features, data = df.gen)
lm.en.gen.preds.int = lm(true_rating ~ type*features*predicted_rating, data = df.gen)

AIC(lm.en.gen.base) # 2395.074
AIC(lm.en.gen.preds) # 1926.128
AIC(lm.en.gen.preds.int) # 1858.691

summary(lm.en.gen.base)
summary(lm.en.gen.preds)
summary(lm.en.gen.preds.int)

gen.preds.en = data.frame(
  ggpredict(lm.en.gen.preds.int, terms = c(
    "predicted_rating[all]",
    "features[all]",
    "type[all]")
  )
)
names(gen.preds.en)[names(gen.preds.en) == 'x'] <- 'modelPrediction'
names(gen.preds.en)[names(gen.preds.en) == 'group'] <- 'FeatureSpace'
names(gen.preds.en)[names(gen.preds.en) == 'facet'] <- 'type'
names(gen.preds.en)[names(gen.preds.en) == 'predicted'] <- 'Rating'


pdf(file = "plots/predictions/lm.en.gender.pdf",  
    width = 7,
    height = 5)
ggplot(data = gen.preds.en, aes(x = modelPrediction, y = Rating)) +
  geom_line(aes(color = FeatureSpace)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = FeatureSpace, fill=FeatureSpace), alpha = 0.4) +
  geom_point(aes(x=predicted_rating, y=true_rating, color=features), size=0.5, data=df.gen) +
  facet_grid(. ~ type) +
  ylim(-1, 1) +
  ggtitle('Prediction relation with gender rating, by features and name type') +
  scale_fill_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred",
    "letters" = "firebrick1",
    "phon_features" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred",
    "letters" = "firebrick1",
    "phon_features" = "deeppink1"
  )) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) 
dev.off()




##### polarity #####
df.pol = df.subset[df.subset$attribute == 'polarity' & df.subset$model == 'elastic net' & df.subset$run =='true mappings', ]

lm.en.pol.base = lm(true_rating ~ type, data = df.pol)
lm.en.pol.preds = lm(true_rating ~ type + predicted_rating + features, data = df.pol)
lm.en.pol.preds.int = lm(true_rating ~ type*features*predicted_rating, data = df.pol)

AIC(lm.en.pol.base) # 268.8424
AIC(lm.en.pol.preds) # 241.2641
AIC(lm.en.pol.preds.int) # 244.1175

summary(lm.en.pol.base)
summary(lm.en.pol.preds)
summary(lm.en.pol.preds.int)

pol.preds.en = data.frame(
  ggpredict(lm.en.pol.preds.int, terms = c(
    "predicted_rating[all]",
    "features[all]",
    "type[all]")
  )
)
names(pol.preds.en)[names(pol.preds.en) == 'x'] <- 'modelPrediction'
names(pol.preds.en)[names(pol.preds.en) == 'group'] <- 'FeatureSpace'
names(pol.preds.en)[names(pol.preds.en) == 'facet'] <- 'type'
names(pol.preds.en)[names(pol.preds.en) == 'predicted'] <- 'Rating'


pdf(file = "plots/predictions/lm.en.polarity.pdf",  
    width = 7,
    height = 5)
ggplot(data = pol.preds.en, aes(x = modelPrediction, y = Rating)) +
  geom_line(aes(color = FeatureSpace)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = FeatureSpace, fill=FeatureSpace), alpha = 0.4) +
  geom_point(aes(x=predicted_rating, y=true_rating, color=features), size=0.5, data=df.pol) +
  facet_grid(. ~ type) +
  ylim(-1, 1) +
  ggtitle('Prediction relation with polarity rating, by features and name type') +
  scale_fill_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred",
    "letters" = "firebrick1",
    "phon_features" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue", 
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred",
    "letters" = "firebrick1",
    "phon_features" = "deeppink1"
  )) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) 
dev.off()

