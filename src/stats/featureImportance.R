library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(reshape2)
library(tidyverse)
library(ggpubr)
library(formattable)

setwd('/Volumes/University/TiU/Research/ResearchTraineeship/2020_21-characters/LangCogn/')

##### LETTERS #####
df.letters = NULL
df.letters.gen = read.csv('output/featureImportance/nn/gender_featureImp_unigrams_M0.csv', header = T, sep = '\t')
df.letters.gen$error = df.letters.gen$true_rating - df.letters.gen$predicted_rating
for (name in unique(df.letters.gen$name)) {
  sub.df = df.letters.gen[df.letters.gen$name == name,]
  base_err = sub.df[sub.df$excluded == 'none', 'error']
  sub.df$diff = sub.df$error - base_err
  df.letters = rbind(df.letters, sub.df)
}

df.letters.age = read.csv('output/featureImportance/nn/age_featureImp_unigrams_M0.csv', header = T, sep = '\t')
df.letters.age$error = df.letters.age$true_rating - df.letters.age$predicted_rating
for (name in unique(df.letters.age$name)) {
  sub.df = df.letters.age[df.letters.age$name == name,]
  base_err = sub.df[sub.df$excluded == 'none', 'error']
  sub.df$diff = sub.df$error - base_err
  df.letters = rbind(df.letters, sub.df)
}

df.letters.pol = read.csv('output/featureImportance/nn/polarity_featureImp_unigrams_M0.csv', header = T, sep = '\t')
df.letters.pol$error = df.letters.pol$true_rating - df.letters.pol$predicted_rating
for (name in unique(df.letters.pol$name)) {
  sub.df = df.letters.pol[df.letters.pol$name == name,]
  base_err = sub.df[sub.df$excluded == 'none', 'error']
  sub.df$diff = sub.df$error + base_err
  df.letters = rbind(df.letters, sub.df)
}

rm(df.letters.age, df.letters.gen, df.letters.pol, sub.df, base_err, name)

df.letters.aggr = df.letters %>%
  select(-feature_vector) %>% 
  mutate_if(is.character, as.factor) %>%
  group_by(excluded, type, attribute) %>%
  summarise(
    type = type,
    attribute = attribute,
    excluded = excluded,
    diff.avg = mean(diff, na.rm = TRUE),
    sd = sd(diff, na.rm = TRUE),
    n = length(diff),
    ci.low = diff.avg - 1.96*sd/sqrt(n),
    ci.high = diff.avg + 1.96*sd/sqrt(n) 
  ) %>%
  distinct(.keep_all = TRUE)


p.lett.gen = ggplot(data = df.letters.aggr[df.letters.aggr$excluded != 'none' & df.letters.aggr$n > 4 & df.letters.aggr$attribute == 'gender',], 
            aes(x=excluded, y=diff.avg)) +
  geom_bar(aes(color=type, fill=type, group=type), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_errorbar(aes(ymin = ci.low, ymax = ci.high, group = type), width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
  labs(title = 'Feature importance - Letters', subtitle = 'gender') +
  scale_y_continuous(limits = c(-1,1), breaks=c(-1, 0, 1), labels = c('male', ' ', 'female'), name ="bias") +
  scale_fill_manual(values = c(
   "madeup" ="turquoise4",
   "real" ="firebrick3",
   "talking" ="darkorange1"
  )) +
  scale_color_manual(values = c(
   "madeup" ="turquoise4",
   "real" ="firebrick3",
   "talking" ="darkorange1"
  )) +
  theme(legend.position = "none",
        axis.title.x=element_blank())


p.lett.age = ggplot(data = df.letters.aggr[df.letters.aggr$excluded != 'none' & df.letters.aggr$n > 4 & df.letters.aggr$attribute == 'age',], 
       aes(x=excluded, y=diff.avg)) +
  geom_bar(aes(color=type, fill=type, group=type), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_errorbar(aes(ymin = ci.low, ymax = ci.high, group = type), width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
  labs(x ="letter", title = '', subtitle = 'age') +
  scale_y_continuous(limits = c(-1,1), breaks=c(-1, 0, 1), labels = c('young', ' ', 'old'), name ="bias") +
  scale_fill_manual(values = c(
   "madeup" ="turquoise4",
   "real" ="firebrick3",
   "talking" ="darkorange1"
  )) +
  scale_color_manual(values = c(
   "madeup" ="turquoise4",
   "real" ="firebrick3",
   "talking" ="darkorange1"
  )) +
  theme(
    legend.justification = c(1, 0),
    legend.position = c(0.8, 0.1),
    legend.direction ="horizontal",
    axis.title.x=element_blank()
  ) +
  guides(fill = guide_legend(title="Name type"),
         color = guide_legend(title="Name type"))


p.lett.pol = ggplot(data = df.letters.aggr[df.letters.aggr$excluded != 'none' & df.letters.aggr$n > 4 & df.letters.aggr$attribute == 'polarity',], 
       aes(x=excluded, y=diff.avg)) +
  geom_bar(aes(color=type, fill=type, group=type), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_errorbar(aes(ymin = ci.low, ymax = ci.high, group = type), width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
  labs(x ="letter", title = '', subtitle = 'polarity') +
  scale_y_continuous(limits = c(-1,1), breaks=c(-1, 0, 1), labels = c('evil', ' ', 'good'), name ="bias") +
  scale_fill_manual(values = c(
   "madeup" ="turquoise4",
   "real" ="firebrick3",
   "talking" ="darkorange1"
  )) +
  scale_color_manual(values = c(
   "madeup" ="turquoise4",
   "real" ="firebrick3",
   "talking" ="darkorange1"
  )) +
  theme(legend.position = "none")
  
  
  

pdf(file = "plots/featureImportance/bars.letters.pdf",
    width = 6, 
    height = 9)
ggarrange(p.lett.gen, p.lett.age, p.lett.pol,
          labels = c("A", "B", "C"),
          ncol = 1, nrow = 3)
dev.off()


##### PHONOLOGICAL FEATURES #####
df.phon = NULL
df.phon.gen = read.csv('output/featureImportance/nn/gender_featureImp_phon_features_M0.csv', header = T, sep = '\t')
df.phon.gen$error = df.phon.gen$true_rating - df.phon.gen$predicted_rating
for (name in unique(df.phon.gen$name)) {
  sub.df = df.phon.gen[df.phon.gen$name == name,]
  base_err = sub.df[sub.df$excluded == 'none', 'error']
  sub.df$diff = sub.df$error - base_err
  df.phon = rbind(df.phon, sub.df)
}

df.phon.age = read.csv('output/featureImportance/nn/age_featureImp_phon_features_M0.csv', header = T, sep = '\t')
df.phon.age$error = df.phon.age$true_rating - df.phon.age$predicted_rating
for (name in unique(df.phon.age$name)) {
  sub.df = df.phon.age[df.phon.age$name == name,]
  base_err = sub.df[sub.df$excluded == 'none', 'error']
  sub.df$diff = sub.df$error - base_err
  df.phon = rbind(df.phon, sub.df)
}

df.phon.pol = read.csv('output/featureImportance/nn/polarity_featureImp_phon_features_M0.csv', header = T, sep = '\t')
df.phon.pol$error = df.phon.pol$true_rating - df.phon.pol$predicted_rating
for (name in unique(df.phon.pol$name)) {
  sub.df = df.phon.pol[df.phon.pol$name == name,]
  base_err = sub.df[sub.df$excluded == 'none', 'error']
  sub.df$diff = sub.df$error - base_err
  df.phon = rbind(df.phon, sub.df)
}


rm(df.phon.age, df.phon.gen, df.phon.pol, sub.df, base_err, name) 

df.phon.aggr = df.phon %>%
  select(-feature_vector) %>% 
  mutate_if(is.character, as.factor) %>%
  group_by(excluded, type, attribute) %>%
  summarise(
    type = type,
    attribute = attribute,
    excluded = excluded,
    diff.avg = mean(diff, na.rm = TRUE),
    sd = sd(diff, na.rm = TRUE),
    n = length(diff),
    ci.low = diff.avg - 1.96*sd/sqrt(n),
    ci.high = diff.avg + 1.96*sd/sqrt(n) 
  ) %>%
  distinct(.keep_all = TRUE)

df.phon.aggr$excluded = mapvalues(df.phon.aggr$excluded,
                                  from = c('Approximants', "Lateral Approximants"),
                                  to = c('Approx.', 'Lat. Approx.'))


p.phon.gen = ggplot(data = df.phon.aggr[df.phon.aggr$excluded != 'none' & df.phon.aggr$n > 4 & df.phon.aggr$attribute == 'gender',], 
       aes(x=excluded, y=diff.avg)) +
  geom_bar(aes(color=type, fill=type, group=type), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_errorbar(aes(ymin = ci.low, ymax = ci.high, group = type), width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
  labs(title = 'Feature importance - phonological features', subtitle = 'gender') +
  scale_y_continuous(limits = c(-0.25, 0.25), breaks=c(-0.25, 0, 0.25), labels = c('male', ' ', 'female'), name ="bias") +
  scale_fill_manual(values = c(
   "madeup" ="turquoise4",
   "real" ="firebrick3",
   "talking" ="darkorange1"
  )) +
  scale_color_manual(values = c(
   "madeup" ="turquoise4",
   "real" ="firebrick3",
   "talking" ="darkorange1"
  )) +
  theme(
    legend.position = "none",
    axis.title.x=element_blank(),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
    )


p.phon.age = ggplot(data = df.phon.aggr[df.phon.aggr$excluded != 'none' & df.phon.aggr$n > 4 & df.phon.aggr$attribute == 'age',], 
       aes(x=excluded, y=diff.avg)) +
  geom_bar(aes(color=type, fill=type, group=type), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_errorbar(aes(ymin = ci.low, ymax = ci.high, group = type), width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
  labs(title = ' ', subtitle = 'age') +
  scale_y_continuous(limits = c(-0.25, 0.25), breaks=c(-0.25, 0, 0.25), labels = c('young', ' ', 'old'), name ="bias") +
  scale_fill_manual(values = c(
   "madeup" ="turquoise4",
   "real" ="firebrick3",
   "talking" ="darkorange1"
  )) +
  scale_color_manual(values = c(
   "madeup" ="turquoise4",
   "real" ="firebrick3",
   "talking" ="darkorange1"
  )) +
  theme(
    legend.justification = c(1, 0),
    axis.title.x=element_blank(),
    legend.position = c(0.8, 0.1),
    legend.direction ="horizontal",
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
  ) +
  guides(fill = guide_legend(title="Name type"),
         color = guide_legend(title="Name type"))


p.phon.pol = ggplot(data = df.phon.aggr[df.phon.aggr$excluded != 'none' & df.phon.aggr$n > 4 & df.phon.aggr$attribute == 'polarity',], 
       aes(x=excluded, y=diff.avg)) +
  geom_bar(aes(color=type, fill=type, group=type), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_errorbar(aes(ymin = ci.low, ymax = ci.high, group = type), width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
  labs(title = ' ', subtitle = 'polarity') +
  scale_y_continuous(limits = c(-0.25, 0.25), breaks=c(-0.25, 0, 0.25), labels = c('evil', ' ', 'good'), name ="bias") +
  scale_fill_manual(values = c(
   "madeup" ="turquoise4",
   "real" ="firebrick3",
   "talking" ="darkorange1"
  )) +
  scale_color_manual(values = c(
   "madeup" ="turquoise4",
   "real" ="firebrick3",
   "talking" ="darkorange1"
  )) +
  theme(
    legend.position ="none",
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
    )


pdf(file = "plots/featureImportance/bars.phonFeatures.pdf",
    width = 6, 
    height = 9)
ggarrange(p.phon.gen, p.phon.age, p.phon.pol,
          labels = c("A", "B", "C"),
          ncol = 1, nrow = 3)
dev.off()


##### NGRAMS #####
df.ngrams = NULL
df.ngrams.gen = read.csv('output/featureImportance/nn/gender_featureImp_embedding_M5.csv', header = T, sep = '\t')
df.ngrams.gen$error = df.ngrams.gen$true_rating - df.ngrams.gen$predicted_rating
for (name in unique(df.ngrams.gen$name)) {
  sub.df = df.ngrams.gen[df.ngrams.gen$name == name,]
  base_err = sub.df[sub.df$excluded == 'none', 'error']
  sub.df$diff = sub.df$error - base_err
  df.ngrams = rbind(df.ngrams, sub.df)
}


df.ngrams.age = read.csv('output/featureImportance/nn/age_featureImp_embedding_M5.csv', header = T, sep = '\t')
df.ngrams.age$error = df.ngrams.age$true_rating - df.ngrams.age$predicted_rating
for (name in unique(df.ngrams.age$name)) {
  sub.df = df.ngrams.age[df.ngrams.age$name == name,]
  base_err = sub.df[sub.df$excluded == 'none', 'error']
  sub.df$diff = sub.df$error - base_err
  df.ngrams = rbind(df.ngrams, sub.df)
}


df.ngrams.pol = read.csv('output/featureImportance/nn/polarity_featureImp_embedding_M5.csv', header = T, sep = '\t')
df.ngrams.pol$error = df.ngrams.pol$true_rating - df.ngrams.pol$predicted_rating
for (name in unique(df.ngrams.pol$name)) {
  sub.df = df.ngrams.pol[df.ngrams.pol$name == name,]
  base_err = sub.df[sub.df$excluded == 'none', 'error']
  sub.df$diff = sub.df$error - base_err
  df.ngrams = rbind(df.ngrams, sub.df)
}

rm(df.ngrams.age, df.ngrams.gen, df.ngrams.pol, sub.df, base_err, name)


df.ngrams.aggr.min = df.ngrams %>%
  select(-feature_vector) %>% 
  filter(diff != 0) %>%
  group_by(name, type, attribute) %>%
  slice_min(order_by = diff)
df.ngrams.aggr.min$extreme = 'min'

df.ngrams.aggr.max = df.ngrams %>%
  select(-feature_vector) %>% 
  filter(diff != 0) %>%
  group_by(name, type, attribute) %>%
  slice_max(order_by = diff)
df.ngrams.aggr.max$extreme = 'max'

df.ngrams.aggr = rbind(df.ngrams.aggr.max, df.ngrams.aggr.min)
rm(df.ngrams.aggr.max, df.ngrams.aggr.min)

df.ngrams.aggr = df.ngrams.aggr %>% 
  mutate(excluded = str_replace(excluded, ">", "#")) %>%
  mutate(excluded = str_replace(excluded, "<", "#"))


ngrams.table = NULL
for (name in unique(df.ngrams.aggr$name)) {
  df.sub = df.ngrams.aggr[df.ngrams.aggr$name == name, ]
  type = unique(df.sub$type)
  female = df.sub[df.sub$attribute == 'gender' & df.sub$extreme == 'min', 'excluded']$excluded
  male = df.sub[df.sub$attribute == 'gender' & df.sub$extreme == 'max', 'excluded']$excluded

  old = df.sub[df.sub$attribute == 'age' & df.sub$extreme == 'min', 'excluded']$excluded
  young = df.sub[df.sub$attribute == 'age' & df.sub$extreme == 'max', 'excluded']$excluded
  
  good = df.sub[df.sub$attribute == 'polarity' & df.sub$extreme == 'min', 'excluded']$excluded
  evil = df.sub[df.sub$attribute == 'polarity' & df.sub$extreme == 'max', 'excluded']$excluded

  if (identical(old, character(0))) {
    old = '-'
    young = '-'
  }
  if (identical(good, character(0))) {
    evil = '-'
    good = '-'
  }
  ngrams.table = rbind(ngrams.table,
                       data.frame(name, type, male, female, young, old, evil, good))
}
rm(df.sub, evil, female, male, good, name, old, type, young)

## notes for interpretation: 
# - max indicates the ngram which, when removed, makes the prediction shift more towards the left pole (male, young, evil)
# - min indicates the ngram which, when removed, makes the prediction shift more towards the right pole (female, old, good)
# max ngrams have thus a bias for the right pole (male, young, evil), and min ngrams have a bias for the left pole (female, old, good)

ngrams.table = ngrams.table[order(ngrams.table$type),]


formattable(ngrams.table, list(
  male = formatter("span", 
                   style = ~style(display = "block",
                                  color = "black",
                                  "border-radius" = "4px",
                                  "padding-right" = "4px",
                                  "color" =  ifelse(`male` == `name`, "dodgerblue", ifelse(nchar(`male`) == 2, "mediumvioletred", "darkgray")),
                                  "font.weight" = ifelse(grepl('#', `male`, fixed = TRUE), "bold", NA))
                   ),
  female = formatter("span", 
                   style = ~style(display = "block",
                                  color = "black",
                                  "border-radius" = "4px",
                                  "padding-right" = "4px",
                                  "color" =  ifelse(`female` == `name`, "dodgerblue", ifelse(nchar(`female`) == 2, "mediumvioletred", "darkgray")),
                                  "font.weight" = ifelse(grepl('#', `female`, fixed = TRUE), "bold", NA))
  ),
  young = formatter("span", 
                   style = ~style(display = "block",
                                  color = "black",
                                  "border-radius" = "4px",
                                  "padding-right" = "4px",
                                  "color" =  ifelse(`young` == `name`, "dodgerblue", ifelse(nchar(`young`) == 2, "mediumvioletred", "darkgray")),
                                  "font.weight" = ifelse(grepl('#', `young`, fixed = TRUE), "bold", NA))
  ),
  old = formatter("span", 
                   style = ~style(display = "block",
                                  color = "black",
                                  "border-radius" = "4px",
                                  "padding-right" = "4px",
                                  "color" =  ifelse(`old` == `name`, "dodgerblue", ifelse(nchar(`old`) == 2, "mediumvioletred", "darkgray")),
                                  "font.weight" = ifelse(grepl('#', `old`, fixed = TRUE), "bold", NA))
  ),
  evil = formatter("span", 
                   style = ~style(display = "block",
                                  color = "black",
                                  "border-radius" = "4px",
                                  "padding-right" = "4px",
                                  "color" =  ifelse(`evil` == `name`, "dodgerblue", ifelse(nchar(`evil`) == 2, "mediumvioletred", "darkgray")),
                                  "font.weight" = ifelse(grepl('#', `evil`, fixed = TRUE), "bold", NA))
  ),
  good = formatter("span", 
                   style = ~style(display = "block",
                                  color = "black",
                                  "border-radius" = "4px",
                                  "padding-right" = "4px",
                                  "color" =  ifelse(`good` == `name`, "dodgerblue", ifelse(nchar(`good`) == 2, "mediumvioletred", "darkgray")),
                                  "font.weight" = ifelse(grepl('#', `good`, fixed = TRUE), "bold", NA))
  )))


##### BIGRAMS #####
df.bigrams = NULL
df.bigrams.gen = read.csv('output/featureImportance/nn/gender_featureImp_embedding_M2.csv', header = T, sep = '\t')
df.bigrams.gen$error = df.bigrams.gen$true_rating - df.bigrams.gen$predicted_rating
for (name in unique(df.bigrams.gen$name)) {
  sub.df = df.bigrams.gen[df.bigrams.gen$name == name,]
  base_err = sub.df[sub.df$excluded == 'none', 'error']
  sub.df$diff = sub.df$error - base_err
  df.bigrams = rbind(df.bigrams, sub.df)
}

df.bigrams.age = read.csv('output/featureImportance/nn/age_featureImp_embedding_M2.csv', header = T, sep = '\t')
df.bigrams.age$error = df.bigrams.age$true_rating - df.bigrams.age$predicted_rating
for (name in unique(df.bigrams.age$name)) {
  sub.df = df.bigrams.age[df.bigrams.age$name == name,]
  base_err = sub.df[sub.df$excluded == 'none', 'error']
  sub.df$diff = sub.df$error - base_err
  df.bigrams = rbind(df.bigrams, sub.df)
}

df.bigrams.pol = read.csv('output/featureImportance/nn/polarity_featureImp_embedding_M2.csv', header = T, sep = '\t')
df.bigrams.pol$error = df.bigrams.pol$true_rating - df.bigrams.pol$predicted_rating
for (name in unique(df.bigrams.pol$name)) {
  sub.df = df.bigrams.pol[df.bigrams.pol$name == name,]
  base_err = sub.df[sub.df$excluded == 'none', 'error']
  sub.df$diff = sub.df$error - base_err
  df.bigrams = rbind(df.bigrams, sub.df)
}


rm(df.bigrams.age, df.bigrams.gen, df.bigrams.pol, sub.df, base_err, name)

df.bigrams.aggr.min = df.bigrams %>%
  select(-feature_vector) %>% 
  filter(excluded != 'none') %>%
  group_by(name, type, attribute) %>%
  slice_min(order_by = diff)
df.bigrams.aggr.min$extreme = 'min'

df.bigrams.aggr.max = df.bigrams %>%
  select(-feature_vector) %>% 
  filter(excluded != 'none') %>%
  group_by(name, type, attribute) %>%
  slice_max(order_by = diff)
df.bigrams.aggr.max$extreme = 'max'

df.bigrams.aggr = rbind(df.bigrams.aggr.max, df.bigrams.aggr.min)
rm(df.bigrams.aggr.max, df.bigrams.aggr.min)

df.bigrams.aggr = df.bigrams.aggr %>% 
  mutate(excluded = str_replace(excluded, ">", "#")) %>%
  mutate(excluded = str_replace(excluded, "<", "#"))

bigrams.table = NULL
for (name in unique(df.bigrams.aggr$name)) {
  print(name)
  df.sub = df.bigrams.aggr[df.bigrams.aggr$name == name, ]
  type = unique(df.sub$type)
  female = df.sub[df.sub$attribute == 'gender' & df.sub$extreme == 'min', 'excluded']$excluded
  male = df.sub[df.sub$attribute == 'gender' & df.sub$extreme == 'max', 'excluded']$excluded
  
  old = df.sub[df.sub$attribute == 'age' & df.sub$extreme == 'min', 'excluded']$excluded
  young = df.sub[df.sub$attribute == 'age' & df.sub$extreme == 'max', 'excluded']$excluded
  
  good = df.sub[df.sub$attribute == 'polarity' & df.sub$extreme == 'min', 'excluded']$excluded
  evil = df.sub[df.sub$attribute == 'polarity' & df.sub$extreme == 'max', 'excluded']$excluded
  
  if (identical(old, character(0))) {
    old = '-'
    young = '-'
  }
  if (identical(good, character(0))) {
    evil = '-'
    good = '-'
  }
  bigrams.table = rbind(bigrams.table,
                        data.frame(name, type, male, female, young, old, evil, good))
}
rm(df.sub, evil, female, male, good, name, old, type, young)

bigrams.table = bigrams.table[order(bigrams.table$type),]


formattable(bigrams.table, list(
  male = formatter("span", 
                   style = ~style(display = "block",
                                  color = "black",
                                  "border-radius" = "4px",
                                  "padding-right" = "4px",
                                  "font.weight" = ifelse(grepl('#', `male`, fixed = TRUE), "bold", NA))
  ),
  female = formatter("span", 
                     style = ~style(display = "block",
                                    color = "black",
                                    "border-radius" = "4px",
                                    "padding-right" = "4px",
                                    "font.weight" = ifelse(grepl('#', `female`, fixed = TRUE), "bold", NA))
  ),
  young = formatter("span", 
                    style = ~style(display = "block",
                                   color = "black",
                                   "border-radius" = "4px",
                                   "padding-right" = "4px",
                                   "font.weight" = ifelse(grepl('#', `young`, fixed = TRUE), "bold", NA))
  ),
  old = formatter("span", 
                  style = ~style(display = "block",
                                 color = "black",
                                 "border-radius" = "4px",
                                 "padding-right" = "4px",
                                 "font.weight" = ifelse(grepl('#', `old`, fixed = TRUE), "bold", NA))
  ),
  evil = formatter("span", 
                   style = ~style(display = "block",
                                  color = "black",
                                  "border-radius" = "4px",
                                  "padding-right" = "4px",
                                  "font.weight" = ifelse(grepl('#', `evil`, fixed = TRUE), "bold", NA))
  ),
  good = formatter("span", 
                   style = ~style(display = "block",
                                  color = "black",
                                  "border-radius" = "4px",
                                  "padding-right" = "4px",
                                  "font.weight" = ifelse(grepl('#', `good`, fixed = TRUE), "bold", NA))
  )))

## notes for interpretation: 
# - max indicates the ngram which, when removed, makes the prediction shift more towards the left pole (male, young, evil)
# - min indicates the ngram which, when removed, makes the prediction shift more towards the right pole (female, old, good)
# max ngrams have thus a bias for the right pole, and min ngrams have a bias for the left pole
