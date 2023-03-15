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
library(reshape2)
library(tidyverse)


setwd('/Volumes/University/TiU/Research/ResearchTraineeship/2020_21-characters/LangCogn/')

names = read.csv('data/avgRatings_annotated.csv', header = T, sep = ',')
names = names %>%
  mutate(name = str_to_lower(name))


folder = 'output/predictions/weat'
attributes = c('polarity', 'age', 'gender')
ngrams = c(0, 2, 5)
lexical = c('True', 'False')

weat_df = NULL
for (attribute in attributes) {
  for (ngram in ngrams) {
    for (lex in lexical) {
      f = paste(paste(folder, paste(attribute, 'associations', paste('ngram', ngram, sep = ''), 'lexical', lex, sep = '_'), sep='/'), 'csv', sep='.')
      to_next = FALSE
      tryCatch(
        expr = {df = read.csv(f, header = T, sep = '\t')
        print(f)
        df = df %>%
          mutate(name = str_to_lower(name))
        df$ngram_size = ngram
        df$attribute = attribute
        df$lexical = lex
        weat_df = rbind(weat_df, df)},
        error = function(e) {
          to_next <<- TRUE
          }
      ) 
      if (to_next) { next }
    }
  }
}

rm(names, df, attribute, folder, ngram, lex, to_next, f)


weat_df.aggr = weat_df %>%
  select(-feature_vector) %>% 
  mutate_if(is.character, as.factor) %>%
  unite('features', ngram_size, lexical, sep='_') %>%
  pivot_wider(names_from = wordset, values_from = cosine_sim) %>%
  group_by(name, attribute, features) %>%
  summarise(
    type = type,
    attribute = attribute,
    true_rating = rating.mean_unit,
    features = features,
    avg_cos_a = mean(a, na.rm = TRUE),
    avg_cos_b = mean(b, na.rm = TRUE),
  ) %>%
  distinct(.keep_all = TRUE)

weat_df.aggr$features = fct_recode(
  weat_df.aggr$features,
  all = '5_True',
  lexical = '0_True',
  bigrams = '2_False',
  ngrams = '5_False'
  )
  
weat_df.aggr$weat_score = weat_df.aggr$avg_cos_b - weat_df.aggr$avg_cos_a

weat_df.aggr[weat_df.aggr$type == 'madeup' & weat_df.aggr$features == 'lexical', 'weat_score'] = NA
weat_df.aggr = weat_df.aggr[complete.cases(weat_df.aggr),]


pdf(file = "plots/weat/smooth.all.exploratory.pdf",
    width = 10, 
    height = 10)
ggplot(data = weat_df.aggr, aes(x=weat_score, y=true_rating)) +
  geom_point(aes(color = features), size = 0.5) +
  geom_smooth(aes(color = features, fill = features), method = 'lm', alpha = 0.3) +
  facet_grid(attribute ~ type) +
  ylim(-1, 1) +
  scale_fill_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred"
  )) +
  scale_color_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred"
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


pdf(file = "plots/weat/density.pdf",
    width = 9, 
    height = 9)
ggplot(data = weat_df.aggr, aes(x=weat_score)) +
  geom_density(aes(color = features, fill = features), alpha = 0.3) +
  facet_grid(attribute ~ type) +
  scale_fill_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred"
  )) +
  scale_color_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred"
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



##### LINEAR MODELS #####
# age
df.age = weat_df.aggr[weat_df.aggr$attribute == 'age',]
lm.age.base = lm(true_rating ~ type, data = df.age)
lm.age.preds = lm(true_rating ~ type + weat_score, data = df.age)
lm.age.preds.int = lm(true_rating ~ type*weat_score*features, data = df.age)

AIC(lm.age.base) # 436.9083
AIC(lm.age.preds) # 397.142
AIC(lm.age.preds.int) # 402.2066 (387.8036 excluding the 3-way interaction and leaving the 2-way interaction between type and weat score)

summary(lm.age.base)
summary(lm.age.preds)
summary(lm.age.preds.int)

age.preds = data.frame(
  ggpredict(lm.age.preds.int, terms = c(
    "weat_score[all]",
    "features[all]",
    "type[all]")
  )
)
names(age.preds)[names(age.preds) == 'x'] <- 'weat_score'
names(age.preds)[names(age.preds) == 'group'] <- 'features'
names(age.preds)[names(age.preds) == 'facet'] <- 'type'

age.preds$features <- factor(age.preds$features, levels = c("lexical", "all", "ngrams", "bigrams"))

age.preds[age.preds$type == 'madeup' & age.preds$features == 'lexical', 'weat_score'] = NA
age.preds = age.preds[complete.cases(age.preds),]


pdf(file = "plots/weat/lines.age.lm.pdf",
    width = 7, 
    height = 5)
ggplot(data = age.preds, aes(x = weat_score, y = predicted)) +
  geom_line(aes(color = features)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = features, fill = features), alpha = 0.3) +
  geom_point(aes(x = weat_score, y = true_rating, color = features), size = 0.5, data=df.age) +
  facet_grid(. ~ type) +
  ylim(-1, 1) +
  scale_fill_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred"
  )) +
  scale_color_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred"
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




# polarity
df.pol = weat_df.aggr[weat_df.aggr$attribute == 'polarity',]

lm.pol.base = lm(true_rating ~ type, data = df.pol)
lm.pol.preds = lm(true_rating ~ type + weat_score, data = df.pol)
lm.pol.preds.int = lm(true_rating ~ type*weat_score*features, data = df.pol)

AIC(lm.pol.base) # 164.4212
AIC(lm.pol.preds) # 161.6974
AIC(lm.pol.preds.int) # 182.5129 (164.9552 when excluding the 3-way interaction, keeping only the 2-way interaction between type and weat_score

summary(lm.pol.base)
summary(lm.pol.preds)
summary(lm.pol.preds.int)

pol.preds = data.frame(
  ggpredict(lm.pol.preds.int, terms = c(
    "weat_score[all]",
    "features[all]",
    "type[all]")
  )
)
names(pol.preds)[names(pol.preds) == 'x'] <- 'weat_score'
names(pol.preds)[names(pol.preds) == 'group'] <- 'features'
names(pol.preds)[names(pol.preds) == 'facet'] <- 'type'

pol.preds$features <- factor(pol.preds$features, levels = c("lexical", "all", "ngrams", "bigrams"))

pol.preds[pol.preds$type == 'madeup' & pol.preds$features == 'lexical', 'weat_score'] = NA
pol.preds = pol.preds[complete.cases(pol.preds),]


pdf(file = "plots/weat/lines.polarity.lm.pdf",
    width = 7, 
    height = 5)
ggplot(data = pol.preds, aes(x = weat_score, y = predicted)) +
  geom_line(aes(color = features)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = features, fill = features), alpha = 0.3) +
  geom_point(aes(x = weat_score, y = true_rating, color = features), size = 0.5, data=df.pol) +
  facet_grid(. ~ type) +
  ylim(-1, 1) +
  scale_fill_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred"
  )) +
  scale_color_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred"
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


# gender
df.gen = weat_df.aggr[weat_df.aggr$attribute == 'gender',]

lm.gen.base = lm(true_rating ~ type, data = df.gen)
lm.gen.preds = lm(true_rating ~ type + weat_score, data = df.gen)
lm.gen.preds.int = lm(true_rating ~ type*weat_score*features, data = df.gen)

AIC(lm.gen.base) # 1479.462
AIC(lm.gen.preds) # 971.4073
AIC(lm.gen.preds.int) # 984.1911   (976.9432 with a 2-way interaction between type and weat score)

summary(lm.gen.base)
summary(lm.gen.preds)
summary(lm.gen.preds.int)

gen.preds = data.frame(
  ggpredict(lm.gen.preds.int, terms = c(
    "weat_score[all]",
    "features[all]",
    "type[all]")
  )
)
names(gen.preds)[names(gen.preds) == 'x'] <- 'weat_score'
names(gen.preds)[names(gen.preds) == 'group'] <- 'features'
names(gen.preds)[names(gen.preds) == 'facet'] <- 'type'

gen.preds$features <- factor(gen.preds$features, levels = c("lexical", "all", "ngrams", "bigrams"))
gen.preds[gen.preds$type == 'madeup' & gen.preds$features == 'lexical', 'weat_score'] = NA
gen.preds = gen.preds[complete.cases(gen.preds),]


pdf(file = "plots/weat/lines.gender.lm.pdf",
    width = 7, 
    height = 5)
ggplot(data = gen.preds, aes(x = weat_score, y = predicted)) +
  geom_line(aes(color = features)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = features, fill = features), alpha = 0.3) +
  geom_point(aes(x = weat_score, y = true_rating, color = features), size = 0.5, data=df.gen) +
  facet_grid(. ~ type) +
  ylim(-1, 1) +
  scale_fill_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred"
  )) +
  scale_color_manual(values = c(
    "lexical" = "dodgerblue3",
    "all" = "slateblue",
    "ngrams" = "mediumorchid4",
    "bigrams" = "mediumvioletred"
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
