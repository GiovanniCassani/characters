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
library(ggbeeswarm)

setwd('/Volumes/University/TiU/Research/ResearchTraineeship/2020_21-characters/LangCogn/')

load("output/MAE.RData")

df.subset = df %>%
  mutate_if(is.character,as.factor) %>%
  filter(features %in% c("M5_lexical_False", "M2_lexical_False", "unigrams", "phon_features", "M0_lexical_True", "M5_lexical_True")) %>%
  filter(run != 'avg') %>%
  mutate(features=recode(features,
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
  mutate(features = fct_relevel(features, c('lexical', 'all', 'ngrams', 'bigrams', 'letters', 'phon_features')))

df.subset = droplevels(df.subset)


##### gender #####
df.gen = df.subset[df.subset$attribute == 'gender', ]

df.gen = df.gen %>%
  group_by(type) %>%
  mutate(base_error = true_rating - mean(true_rating),
         absolute_base_error = abs(base_error))

df.gen$err_diff = abs(df.gen$absolute_base_error - df.gen$absolute_error)

pdf(file = "plots/MAE/violin.gender.pdf",   
    width = 12, 
    height = 7)
ggplot(data = df.gen, aes(x = features, y=absolute_error)) +
  geom_violin(aes(color = features, fill=features, alpha = run), position = position_dodge(width = 0.7)) +
  geom_quasirandom(aes(color = features, group = run), size=0.25, dodge.width = 0.7) +
  ggtitle('Absolute error, by feature - Gender') +
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
  scale_alpha_manual(values = c(
    'true mappings' = 0.1,
    'random mappings' = 0.9
  )) +
  facet_grid(type ~ model) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'horizontal', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) +
  guides(alpha = guide_legend(override.aes = list(fill = c('darkgrey','darkgrey'))))
dev.off()


### NEURAL NETWORKS
lm.gen.nn.err.base = lm(absolute_error ~ 1, 
                     data = df.gen[df.gen$model == 'neural network' & df.gen$run =='true mappings', ])
AIC(lm.gen.nn.err.base)  # 505.5211
lm.gen.nn.err.type = lm(absolute_error ~ type, 
                     data = df.gen[df.gen$model == 'neural network' & df.gen$run =='true mappings', ])
AIC(lm.gen.nn.err.type)  # 492.683
lm.gen.nn.err.feat = lm(absolute_error ~ features, 
                     data = df.gen[df.gen$model == 'neural network' & df.gen$run =='true mappings', ])
AIC(lm.gen.nn.err.feat)  # 241.7275
lm.gen.nn.err.comb = lm(absolute_error ~ type + features, 
                     data = df.gen[df.gen$model == 'neural network' & df.gen$run =='true mappings', ])
AIC(lm.gen.nn.err.comb)  # 223.9503
lm.gen.nn.err.int = lm(absolute_error ~ type*features, 
                    data = df.gen[df.gen$model == 'neural network' & df.gen$run =='true mappings', ])
AIC(lm.gen.nn.err.int)  # 49.86615
summary(lm.gen.nn.err.int)
# the interaction between name type and feature space improves the prediction of absolute errors

gen.preds.nn.err = data.frame(
  ggpredict(lm.gen.nn.err.int, terms = c(
    "type[all]",
    "features[all]"
    )
  )
)
names(gen.preds.nn.err)[names(gen.preds.nn.err) == 'x'] <- 'type'
names(gen.preds.nn.err)[names(gen.preds.nn.err) == 'group'] <- 'FeatureSpace'
names(gen.preds.nn.err)[names(gen.preds.nn.err) == 'predicted'] <- 'AbsoluteError'

pdf(file = "plots/MAE/bars.gender.nn.lm.pdf",   # The directory you want to save the file in
    width = 7, # The width of the plot in inches
    height = 5)
ggplot(data = gen.preds.nn.err, aes(x=type, y=AbsoluteError)) +
  geom_bar(aes(fill=FeatureSpace, color=FeatureSpace), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_point(aes(x=type, y=absolute_error, color=features), size=0.5, data=df.gen[df.gen$model == 'nn' & df.gen$run =='true', ], position=position_dodge(0.5)) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high, group = FeatureSpace), 
                width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
  ggtitle('MAE (ANN predictions) - Gender') +
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
    legend.justification = c(1, 0),
    legend.position = "bottom",
    legend.direction = "horizontal",
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10)
  )
dev.off()

### ELASTICNET
lm.gen.en.err.base = lm(absolute_error ~ 1, 
                     data = df.gen[df.gen$model == 'elastic net' & df.gen$run =='true mappings', ])
AIC(lm.gen.en.err.base)  # 728.0964
lm.gen.en.err.type = lm(absolute_error ~ type, 
                     data = df.gen[df.gen$model == 'elastic net' & df.gen$run =='true mappings', ])
AIC(lm.gen.en.err.type)  # 721.2681
lm.gen.en.err.feat = lm(absolute_error ~ features, 
                     data = df.gen[df.gen$model == 'elastic net' & df.gen$run =='true mappings', ])
AIC(lm.gen.en.err.feat)  # 626.0837
lm.gen.en.err.comb = lm(absolute_error ~ type + features, 
                     data = df.gen[df.gen$model == 'elastic net' & df.gen$run =='true mappings', ])
AIC(lm.gen.en.err.comb)  # 618.0584
lm.gen.en.err.int = lm(absolute_error ~ type*features, 
                    data = df.gen[df.gen$model == 'elastic net' & df.gen$run =='true mappings', ])
AIC(lm.gen.en.err.int)  # 542.3606
summary(lm.gen.en.err.int)
# the interaction between name type and feature space improves the prediction of absolute errors

gen.preds.en.err = data.frame(
  ggpredict(lm.gen.en.err.int, terms = c(
    "type[all]",
    "features[all]"
  )
  )
)
names(gen.preds.en.err)[names(gen.preds.en.err) == 'x'] <- 'type'
names(gen.preds.en.err)[names(gen.preds.en.err) == 'group'] <- 'FeatureSpace'
names(gen.preds.en.err)[names(gen.preds.en.err) == 'predicted'] <- 'AbsoluteError'

pdf(file = "plots/MAE/bars.gender.en.lm.pdf",   # The directory you want to save the file in
    width = 7, # The width of the plot in inches
    height = 5)
ggplot(data = gen.preds.en.err, aes(x=type, y=AbsoluteError)) +
  geom_bar(aes(fill=FeatureSpace, color=FeatureSpace), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_point(aes(x=type, y=absolute_error, color=features), size=0.5, data=df.gen[df.gen$model == 'nn' & df.gen$run =='true', ], position=position_dodge(0.5)) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high, group = FeatureSpace), 
                width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
  ggtitle('MAE (ElasticNet predictions) - Gender') +
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
    legend.justification = c(1, 0),
    legend.position = "bottom",
    legend.direction = "horizontal",
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10)
  )
dev.off()




##### age #####
df.age = df.subset[df.subset$attribute == 'age', ]

df.age = df.age %>%
  group_by(type) %>%
  mutate(base_error = true_rating - mean(true_rating),
         absolute_base_error = abs(base_error))

df.age$err_diff = abs(df.age$absolute_base_error - df.age$absolute_error)

pdf(file = "plots/MAE/violin.age.pdf",   # The directory you want to save the file in
    width = 12, # The width of the plot in inches
    height = 7)
ggplot(data = df.age, aes(x = features, y=absolute_error)) +
  geom_violin(aes(color = features, fill=features, alpha = run), position = position_dodge(width = 0.7)) +
  geom_quasirandom(aes(color = features, group = run), size=0.25, dodge.width = 0.7) +
  ggtitle('Absolute error, by feature - Age') +
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
  scale_alpha_manual(values = c(
    'true mappings' = 0.1,
    'random mappings' = 0.9
  )) +
  facet_grid(type ~ model) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'horizontal', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) +
  guides(alpha = guide_legend(override.aes = list(fill = c('darkgrey','darkgrey'))))
dev.off()


### NEURAL NETWORKS
lm.age.nn.err.base = lm(absolute_error ~ 1, data = df.age[df.age$model == 'neural network' & df.age$run =='true mappings', ])
AIC(lm.age.nn.err.base)  # -334.5502
lm.age.nn.err.type = lm(absolute_error ~ type, data = df.age[df.age$model == 'neural network' & df.age$run =='true mappings', ])
AIC(lm.age.nn.err.type)  # -332.4039
lm.age.nn.err.feat = lm(absolute_error ~ features, data = df.age[df.age$model == 'neural network' & df.age$run =='true mappings', ])
AIC(lm.age.nn.err.feat)  # -436.5781
lm.age.nn.err.comb = lm(absolute_error ~ type + features, data = df.age[df.age$model == 'neural network' & df.age$run =='true mappings', ])
AIC(lm.age.nn.err.comb)  # -434.7471
lm.age.nn.err.int = lm(absolute_error ~ type*features, data = df.age[df.age$model == 'neural network' & df.age$run =='true mappings', ])
AIC(lm.age.nn.err.int)  # -456.4838
summary(lm.age.nn.err.int)
# the interaction between name type and feature space improves the prediction of absolute errors

age.preds.nn.err = data.frame(
  ggpredict(lm.age.nn.err.int, terms = c(
    "type[all]",
    "features[all]"
  )
  )
)
names(age.preds.nn.err)[names(age.preds.nn.err) == 'x'] <- 'type'
names(age.preds.nn.err)[names(age.preds.nn.err) == 'group'] <- 'FeatureSpace'
names(age.preds.nn.err)[names(age.preds.nn.err) == 'predicted'] <- 'AbsoluteError'

pdf(file = "plots/MAE/bars.age.nn.lm.pdf",   # The directory you want to save the file in
    width = 7, # The width of the plot in inches
    height = 5)
ggplot(data = age.preds.nn.err, aes(x=type, y=AbsoluteError)) +
  geom_bar(aes(fill=FeatureSpace, color=FeatureSpace), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_point(aes(x=type, y=absolute_error, color=features), size=0.5, data=df.age[df.age$model == 'neural networks' & df.age$run =='true mappings', ], position=position_dodge(0.5)) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high, group = FeatureSpace), 
                width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
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
  ggtitle('MAE (ANN predictions) - Age') +
  theme(
    legend.justification = c(1, 0),
    legend.position = "bottom",
    legend.direction = "horizontal",
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10)
  )
dev.off()

### ELASTICNET
lm.age.en.err.base = lm(absolute_error ~ 1, data = df.age[df.age$model == 'elastic net' & df.age$run =='true mappings', ])
AIC(lm.age.en.err.base)  # -232.7501
lm.age.en.err.type = lm(absolute_error ~ type, data = df.age[df.age$model == 'elastic net' & df.age$run =='true mappings', ])
AIC(lm.age.en.err.type)  # -230.1175
lm.age.en.err.feat = lm(absolute_error ~ features, data = df.age[df.age$model == 'elastic net' & df.age$run =='true mappings', ])
AIC(lm.age.en.err.feat)  # -233.5251
lm.age.en.err.comb = lm(absolute_error ~ type + features, data = df.age[df.age$model == 'elastic net' & df.age$run =='true mappings', ])
AIC(lm.age.en.err.comb)  # -230.9133
lm.age.en.err.int = lm(absolute_error ~ type*features, data = df.age[df.age$model == 'elastic net' & df.age$run =='true mappings', ])
AIC(lm.age.en.err.int)  # -216.49
summary(lm.age.en.err.int)
# the interaction between name type and feature space improves the prediction of absolute errors

age.preds.en.err = data.frame(
  ggpredict(lm.age.en.err.int, terms = c(
    "type[all]",
    "features[all]"
  )
  )
)
names(age.preds.en.err)[names(age.preds.en.err) == 'x'] <- 'type'
names(age.preds.en.err)[names(age.preds.en.err) == 'group'] <- 'FeatureSpace'
names(age.preds.en.err)[names(age.preds.en.err) == 'predicted'] <- 'AbsoluteError'

pdf(file = "plots/MAE/bars.age.en.lm.pdf",   # The directory you want to save the file in
    width = 7, # The width of the plot in inches
    height = 5)
ggplot(data = age.preds.en.err, aes(x=type, y=AbsoluteError)) +
  geom_bar(aes(fill=FeatureSpace, color=FeatureSpace), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_point(aes(x=type, y=absolute_error, color=features), size=0.5, data=df.age[df.age$model == 'elastic nets' & df.age$run =='true mappings', ], position=position_dodge(0.5)) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high, group = FeatureSpace), 
                width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
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
  ggtitle('MAE (ElasticNet predictions) - Age') +
  theme(
    legend.justification = c(1, 0),
    legend.position = "bottom",
    legend.direction = "horizontal",
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10)
  )
dev.off()





##### polarity #####
df.pol = df.subset[df.subset$attribute == 'polarity', ]

df.pol = df.pol %>%
  group_by(type) %>%
  mutate(base_error = true_rating - mean(true_rating),
         absolute_base_error = abs(base_error))

df.pol$err_diff = abs(df.pol$absolute_base_error - df.pol$absolute_error)

pdf(file = "plots/MAE/violin.polarity.pdf",
    width = 12, 
    height = 7)
ggplot(data = df.pol, aes(x = features, y=absolute_error)) +
  geom_violin(aes(color = features, fill=features, alpha = run), position = position_dodge(width = 0.7)) +
  geom_quasirandom(aes(color = features, group = run), size=0.25, dodge.width = 0.7) +
  ggtitle('Absolute error, by feature - Polarity') +
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
  scale_alpha_manual(values = c(
    'true mappings' = 0.1,
    'random mappings' = 0.9
  )) +
  facet_grid(type ~ model) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'horizontal', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) +
  guides(alpha = guide_legend(override.aes = list(fill = c('darkgrey','darkgrey'))))
dev.off()


### NEURAL NETWORKS
lm.pol.nn.err.base = lm(absolute_error ~ 1, data = df.pol[df.pol$model == 'neural network' & df.pol$run =='true mappings', ])
AIC(lm.pol.nn.err.base)  # -184.7307
lm.pol.nn.err.type = lm(absolute_error ~ type, data = df.pol[df.pol$model == 'neural network' & df.pol$run =='true mappings', ])
AIC(lm.pol.nn.err.type)  # -200.7788
lm.pol.nn.err.feat = lm(absolute_error ~ features, data = df.pol[df.pol$model == 'neural network' & df.pol$run =='true mappings', ])
AIC(lm.pol.nn.err.feat)  # -244.091
lm.pol.nn.err.comb = lm(absolute_error ~ type + features, data = df.pol[df.pol$model == 'neural network' & df.pol$run =='true mappings', ])
AIC(lm.pol.nn.err.comb)  # -264.3089
lm.pol.nn.err.int = lm(absolute_error ~ type*features, data = df.pol[df.pol$model == 'neural network' & df.pol$run =='true mappings', ])
AIC(lm.pol.nn.err.int)  # -267.47

summary(lm.pol.nn.err.int)
# the interaction between name type and feature space improves the prediction of absolute errors

pol.preds.nn.err = data.frame(
  ggpredict(lm.pol.nn.err.int, terms = c(
    "type[all]",
    "features[all]"
  )
  )
)
names(pol.preds.nn.err)[names(pol.preds.nn.err) == 'x'] <- 'type'
names(pol.preds.nn.err)[names(pol.preds.nn.err) == 'group'] <- 'FeatureSpace'
names(pol.preds.nn.err)[names(pol.preds.nn.err) == 'predicted'] <- 'AbsoluteError'

pdf(file = "plots/MAE/bars.polarity.nn.lm.pdf",   # The directory you want to save the file in
    width = 7, # The width of the plot in inches
    height = 5)
ggplot(data = pol.preds.nn.err, aes(x=type, y=AbsoluteError)) +
  geom_bar(aes(fill=FeatureSpace, color=FeatureSpace), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_point(aes(x=type, y=absolute_error, color=features), size=0.5, data=df.pol[df.pol$run == 'true mappings' & df.pol$model == 'neural networks',], position=position_dodge(0.5)) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high, group = FeatureSpace), 
                width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
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
  ggtitle('MAE (ANN) - Polarity') +
  theme(
    legend.justification = c(1, 0),
    legend.position = "bottom",
    legend.direction = "horizontal",
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10)
  )
dev.off()


### ELASTICNET
lm.pol.en.err.base = lm(absolute_error ~ 1, data = df.pol[df.pol$model == 'elastic net' & df.pol$run =='true mappings', ])
AIC(lm.pol.en.err.base)  # -138.4408
lm.pol.en.err.type = lm(absolute_error ~ type, data = df.pol[df.pol$model == 'elastic net' & df.pol$run =='true mappings', ])
AIC(lm.pol.en.err.type)  # -143.3283
lm.pol.en.err.feat = lm(absolute_error ~ features, data = df.pol[df.pol$model == 'elastic net' & df.pol$run =='true mappings', ])
AIC(lm.pol.en.err.feat)  # -131.3002
lm.pol.en.err.comb = lm(absolute_error ~ type + features, data = df.pol[df.pol$model == 'elastic net' & df.pol$run =='true mappings', ])
AIC(lm.pol.en.err.comb)  # -136.256
lm.pol.en.err.int = lm(absolute_error ~ type*features, data = df.pol[df.pol$model == 'elastic net' & df.pol$run =='true mappings', ])
AIC(lm.pol.en.err.int)  # -117.6789

summary(lm.pol.en.err.int)
# the interaction between name type and feature space improves the prediction of absolute errors

pol.preds.en.err = data.frame(
  ggpredict(lm.pol.en.err.int, terms = c(
    "type[all]",
    "features[all]"
  )
  )
)
names(pol.preds.en.err)[names(pol.preds.en.err) == 'x'] <- 'type'
names(pol.preds.en.err)[names(pol.preds.en.err) == 'group'] <- 'FeatureSpace'
names(pol.preds.en.err)[names(pol.preds.en.err) == 'predicted'] <- 'AbsoluteError'

pdf(file = "plots/MAE/bars.polarity.en.lm.pdf",   # The directory you want to save the file in
    width = 7, # The width of the plot in inches
    height = 5)
ggplot(data = pol.preds.en.err, aes(x=type, y=AbsoluteError)) +
  geom_bar(aes(fill=FeatureSpace, color=FeatureSpace), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_point(aes(x=type, y=absolute_error, color=features), size=0.5, data=df.pol[df.pol$run == 'true mappings' & df.pol$model == 'elastic nets',], position=position_dodge(0.5)) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high, group = FeatureSpace), 
                width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
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
  ggtitle('MAE (ElasticNet predictions) - Polarity') +
  theme(
    legend.justification = c(1, 0),
    legend.position = "bottom",
    legend.direction = "horizontal",
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10)
  )
dev.off()

