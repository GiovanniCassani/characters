library(plyr)
library(ggExtra)
library(lme4)
library(readxl)
library(tidyverse)
library(reshape2)
library(gridExtra)
library(lmerTest)
library(MuMIn)
library(ggplot2)
library(plotly)
library(htmlwidgets)
library(dplyr)


substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}

geom_point2 <- function (...) {
  GeomPoint2$new(...)
}

setwd('/Volumes/University/TiU/Research/ResearchTraineeship/2020_21-characters/')

##### PREPROCESSING RATINGS #####
df.ratings = read_excel('Data/AronThesis/ratings.xlsx')[-1,]
names = read.csv('Data/AronThesis/names_annotated.csv')
  
# filter raters who did not unambiguously indicate they only speak English fluently (10 subjects dropped)
df.ratings = df.ratings[df.ratings$Speak_languages == 'No',]

# clean-up the name Millicent which was inadvertently included twice in the survey
df.ratings <- df.ratings[, c(5, 12:15, 19:313, 316:355, 358:383)]

# turn survey duration and all ratings to numeric, and demographics to factor
df.ratings[,c(1,6:length(colnames(df.ratings)))] <- lapply(df.ratings[,c(1,6:length(colnames(df.ratings)))],as.numeric)
df.ratings[,2:5] <- lapply(df.ratings[,2:5],as.factor)

# get rid of the '_1' at the end of column names
new_colnames = vector()
for (i in 1:length(colnames(df.ratings))) {
  if (substrRight(colnames(df.ratings)[i], 2) == '_1') {
    new_name = substr(colnames(df.ratings)[i],1,nchar(colnames(df.ratings)[i])-2)
    if (substrRight(new_name, 1) == ' ') {
      new_name = substr(new_name,1,nchar(new_name)-1)
    }
    new_colnames[i] = new_name
  } else {
    new_colnames[i] = colnames(df.ratings)[i]
  }
}
colnames(df.ratings) = new_colnames
rm(new_colnames, i, new_name)

# get stats about experiment duration
mean(df.ratings$`Duration (in seconds)`)
# 475.5241
quantile(df.ratings$`Duration (in seconds)`, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = T)
#     0%     25%     50%     75%    100% 
# 181.00  314.25  403.00  551.00 1973.00 
hist(df.ratings$`Duration (in seconds)`, breaks = 25)

ratings.full = melt(df.ratings, 
                    id.vars=c('Duration (in seconds)', 'Prolific_ID', 'Gender', 'Age', 'Books'), 
                    measure.vars=6:366, variable.name="name_attr", value.name = 'rating')
ratings.full = ratings.full[complete.cases(ratings.full), ]
ratings.full = ratings.full %>% separate(name_attr, c("name", "attribute"), sep = '_')

quants <- c(0.1, 0.9)
# quants <- c(0, 1)
cut_offs = apply( df.ratings[8:length(colnames(df.ratings))], 2 , quantile , probs = quants , na.rm = TRUE )

for (colname in colnames(cut_offs)) {
  for (j in 1:nrow(df.ratings[,colname])) {
    if (!is.na(df.ratings[j,colname])) {
      if ((df.ratings[j,colname] < cut_offs[1,colname]) || (df.ratings[j,colname] > cut_offs[2,colname])) {
        df.ratings[j,colname] = NA
      }
    }
  }
}
rm(cut_offs, quants, j, colname)

ratings = melt(df.ratings, 
               id.vars=c('Duration (in seconds)', 'Prolific_ID', 'Gender', 'Age', 'Books'), 
               measure.vars=6:366, variable.name="name_attr", value.name = 'rating')
ratings = ratings[complete.cases(ratings), ]
ratings = ratings %>% separate(name_attr, c("name", "attribute"), sep = '_')
ratings$name = as.factor(ratings$name)
ratings$attribute = as.factor(ratings$attribute)
ratings$Age.coarse = fct_lump_n(ratings$Age, 2, other_level = '40+')

rated_names = levels(ratings$name)
length(rated_names)

annotated_names = levels(factor(names$name))
length(annotated_names)

annotated_names[!(annotated_names %in% rated_names)]
rated_names[!(rated_names %in% annotated_names)]

ratings = merge(ratings, names, by = 'name')


ratings.gender = ratings[ratings$attribute == 'gender',]
ratings.age = ratings[ratings$attribute == 'age',]
ratings.valence = ratings[ratings$attribute == 'valence',]


##### PLOT RATINGS BY NAME WITH LOESS FIT #####
ratings.gender$name = fct_reorder(ratings.gender$name, ratings.gender$rating, mean, na.rm = T)
ggplot(data = ratings.gender, aes(name, rating)) +
  geom_point(position = position_jitter(width=0.25), colour = 'darkgrey', size=1.5) +
  geom_smooth(aes(as.numeric(name), rating), method = 'loess', colour = 'black', fill = 'steelblue') +
  labs(title = 'Ratings per name', subtitle = 'gender') +
  theme(axis.text.x = element_text(angle = 90)) +
  facet_grid(type ~ .)


ratings.age$name = fct_reorder(ratings.age$name, ratings.age$rating, mean, na.rm = T)
ggplot(data = ratings.age, aes(name, rating)) +
  geom_point(position = position_jitter(width=0.25), colour = 'darkgrey', size=1.5) +
  geom_smooth(aes(as.numeric(name), rating), method = 'loess', colour = 'black', fill = 'steelblue') +
  labs(title = '', subtitle = 'age') +
  theme(axis.text.x = element_text(angle = 90))  +
  facet_grid(type ~ .)

ratings.valence$name = fct_reorder(ratings.valence$name, ratings.valence$rating, mean, na.rm = T)
ggplot(data = ratings.valence, aes(name, rating)) +
  geom_point(position = position_jitter(width=0.25), colour = 'darkgrey', size=1.5) +
  geom_smooth(aes(as.numeric(name), rating), method = 'loess', colour = 'black', fill = 'steelblue') +
  labs(title = '', subtitle = 'valence') +
  theme(axis.text.x = element_text(angle = 90))  +
  facet_grid(type ~ .)


leave_out_Books = 'Prefer not to say'
leave_out_Age.coarse = ''
leave_out_Gender = 'Non-binary / third gender'


ggplot(data = ratings.gender[ratings.gender$Books != leave_out_Books,], aes(name, rating)) +
  geom_point(aes(fill = Books, color = Books), position = position_dodge(0.25), size=0.75)  +
  geom_smooth(aes(as.numeric(name), rating, color = Books, fill = Books), method = 'loess', alpha = 0.2, size = 2) +
  geom_smooth(aes(as.numeric(name), rating), method = 'loess', alpha = 0.2, size = 2, colour = 'black', fill = NA) +
  facet_grid(type ~ .) +
  labs(title = "Rating by number of books read", 
       subtitle = 'Books', 
       x = '', 
       color = 'Books', 
       fill = 'Books') +
  theme(legend.position = "bottom", legend.direction = 'horizontal',
        axis.text.x = element_text(angle = 90))

ggplot(data = ratings.age[ratings.age$Books != leave_out_Books,], aes(name, rating)) +
  geom_point(aes(fill = Books, color = Books), position = position_dodge(0.25), size=0.75)  +
  geom_smooth(aes(as.numeric(name), rating, color = Books, fill = Books), method = 'loess', alpha = 0.2, size = 2) +
  geom_smooth(aes(as.numeric(name), rating), method = 'loess', alpha = 0.2, size = 2, colour = 'black', fill = NA) +
  facet_grid(type ~ .) +
  labs(title = "Rating by number of books read", 
       subtitle = 'Age', 
       x = '', 
       color = 'Books', 
       fill = 'Books') +
  theme(legend.position = "bottom", legend.direction = 'horizontal',
        axis.text.x = element_text(angle = 90))

ggplot(data = ratings.valence[ratings.valence$Books != leave_out_Books,], aes(name, rating)) +
  geom_point(aes(fill = Books, color = Books), position = position_dodge(0.25), size=0.75)  +
  geom_smooth(aes(as.numeric(name), rating, color = Books, fill = Books), method = 'loess', alpha = 0.2, size = 2) +
  geom_smooth(aes(as.numeric(name), rating), method = 'loess', alpha = 0.2, size = 2, colour = 'black', fill = NA) +
  facet_grid(type ~ .) +
  labs(title = "Rating by number of books read", 
       subtitle = 'Valence', 
       x = '', 
       color = 'Books', 
       fill = 'Books') +
  theme(legend.position = "bottom", legend.direction = 'horizontal',
        axis.text.x = element_text(angle = 90))



##### CHECK SYSTEMATIC DIFFERENCES IN RATINGS ACROSS DEMOGRAPHICS #####

mosaicplot(~Age.coarse + Gender + Books, data = ratings)

table(df.ratings$Gender)
table(df.ratings$Age)
table(df.ratings$Books)


lm.rating.age.real = lmer(rating ~ Gender + Age.coarse + Books + (1|Prolific_ID) + (1|name), 
                          data = ratings.age[ratings.age$Gender != leave_out_Gender & 
                                               ratings.age$Books != leave_out_Books &
                                               ratings.age$type == 'real', ])
summary(lm.rating.age.real)
# no reliable differences in ratings of real names with target attribute age

lm.rating.age.madeup = lmer(rating ~ Gender + Age.coarse + Books + (1|Prolific_ID) + (1|name), 
                            data = ratings.age[ratings.age$Gender != leave_out_Gender & 
                                                 ratings.age$Books != leave_out_Books &
                                                 ratings.age$type == 'made-up', ])
summary(lm.rating.age.madeup)
# older raters (40+) tend to give lower ratings for age about made-up names (names are perceived as generally younger).

lm.rating.age.talking = lmer(rating ~ Gender + Age.coarse + Books + (1|Prolific_ID) + (1|name), 
                             data = ratings.age[ratings.age$Gender != leave_out_Gender & 
                                                  ratings.age$Books != leave_out_Books &
                                                  ratings.age$type == 'talking', ])
summary(lm.rating.age.talking)
# no reliable differences in ratings of real names with target attribute age



lm.rating.gender.real = lmer(rating ~ Gender + Age.coarse + Books + (1|Prolific_ID) + (1|name), 
                             data = ratings.gender[ratings.gender$Gender != leave_out_Gender & 
                                                     ratings.gender$Books != leave_out_Books &
                                                     ratings.gender$type == 'real', ])
summary(lm.rating.gender.real)
# older raters (40+) give higher ratings of real name for gender (names are perceived as more female)

lm.rating.gender.madeup = lmer(rating ~ Gender + Age.coarse + Books + (1|Prolific_ID) + (1|name), 
                               data = ratings.gender[ratings.gender$Gender != leave_out_Gender & 
                                                       ratings.gender$Books != leave_out_Books &
                                                       ratings.gender$type == 'made-up', ])
summary(lm.rating.gender.madeup)
# male raters tend to perceive made-up names as more female
# who read 1 book perceives made-up names as more male

lm.rating.gender.talking = lmer(rating ~ Gender + Age.coarse + Books + (1|Prolific_ID) + (1|name), 
                                data = ratings.gender[ratings.gender$Gender != leave_out_Gender & 
                                                        ratings.gender$Books != leave_out_Books &
                                                        ratings.gender$type == 'talking', ])
summary(lm.rating.gender.talking)
# no differences in how talking names are perceived along the gender dimension



lm.rating.valence.real = lmer(rating ~ Gender + Age.coarse + Books + (1|Prolific_ID) + (1|name), 
                             data = ratings.valence[ratings.valence$Gender != leave_out_Gender & 
                                                     ratings.valence$Books != leave_out_Books &
                                                     ratings.valence$type == 'real', ])
summary(lm.rating.valence.real)
# no difference in how raters perceive valence of real names

lm.rating.valence.madeup = lmer(rating ~ Gender + Age.coarse + Books + (1|Prolific_ID) + (1|name), 
                               data = ratings.valence[ratings.valence$Gender != leave_out_Gender & 
                                                       ratings.valence$Books != leave_out_Books &
                                                       ratings.valence$type == 'made-up', ])
summary(lm.rating.valence.madeup)
# no difference in how raters perceive valence of madeup names

lm.rating.valence.talking = lmer(rating ~ Gender + Age.coarse + Books + (1|Prolific_ID) + (1|name), 
                                data = ratings.valence[ratings.valence$Gender != leave_out_Gender & 
                                                        ratings.valence$Books != leave_out_Books &
                                                        ratings.valence$type == 'talking', ])
summary(lm.rating.valence.talking)
# no difference in how raters perceive valence of madeup names


##### SUMMARY MEASURES ####

ratings$type = mapvalues(ratings$type, from = 'made-up', to = 'madeup')
names$type = mapvalues(names$type, from = 'made-up', to = 'madeup')
ratings$attribute = mapvalues(ratings$attribute, from = 'valence', to = 'polarity')
ratings$polarity = mapvalues(ratings$polarity, from = 'bad', to = 'evil')
names$polarity = mapvalues(names$polarity, from = 'bad', to = 'evil')

ratings.aggr = do.call(data.frame, aggregate(
  rating ~ name + type + attribute, ratings, 
  function(x) c(mean = mean(x), sd = sd(x), n = length(x), median = median(x), mad=mad(x), iqr=IQR(x),
                q = quantile(x, probs = 0.25), q = quantile(x, probs = 0.75))
))

df = merge(names, ratings.aggr, by = c('name', 'type'), all.y = T) 
write.csv(df, file = "Data/AronThesis/avgRatings_annotated.csv", row.names = FALSE)


df.item = merge(names, dplyr::select(ratings, Prolific_ID, name, attribute, rating, type), by = c('name', 'type'), all.y = T) 
write.csv(df.item, file = "Data/AronThesis/Ratings_annotated.csv", row.names = FALSE)


##### CORRELATIONS ####

df.gender_polarity = df[df$attribute != 'age', ] %>% drop_na(type)
df.gender_age = df[df$attribute != 'polarity', ] %>% drop_na(type)

df.gender_polarity = reshape(
  dplyr::select(df.gender_polarity, name, type, gender, polarity, attribute, rating.median, rating.q.25., rating.q.75., rating.mean, rating.sd),
  idvar = c("name", "type", "gender", "polarity"), timevar = "attribute", direction = "wide"
  )
df.gender_polarity = df.gender_polarity[complete.cases(df.gender_polarity),]
df.gender_polarity$type = factor(df.gender_polarity$type)

df.gender_age = reshape(
  dplyr::select(df.gender_age, name, type, gender, age, attribute, rating.median, rating.q.25., rating.q.75., rating.mean, rating.sd),
  idvar = c("name", "type", "gender", "age"), timevar = "attribute", direction = "wide"
)
df.gender_age = df.gender_age[complete.cases(df.gender_age),]
df.gender_age$type = factor(df.gender_age$type)

gender_age_avg = ggplot(data = df.gender_age, 
                        aes(rating.mean.age, rating.mean.gender, color = age, fill = age, shape = gender, text = name)) +
  geom_point(size = 1.5) +
  geom_errorbar(aes(ymin = rating.mean.gender - rating.sd.gender, ymax = rating.mean.gender + rating.sd.gender, linetype = gender), 
                width = 0) +
  geom_errorbarh(aes(xmin = rating.mean.age - rating.sd.age, xmax = rating.mean.age + rating.sd.age, color = age), 
                 height = 0) +
  scale_x_continuous(name = 'Age (mean)', 
                     breaks = c(min(df.gender_age$rating.mean.age), max(df.gender_age$rating.mean.age)),
                     labels = c('young', 'old')) + 
  scale_y_continuous(name = 'Gender (mean)', 
                     breaks = c(min(df.gender_age$rating.mean.gender), max(df.gender_age$rating.mean.gender)),
                     labels = c('male', 'female')) + 
  facet_grid(. ~ type) 

gender_age_med = ggplot(data = df.gender_age, 
                        aes(rating.median.age, rating.median.gender, color = age, fill = age, shape = gender, text = name)) +
  geom_point(size = 1.5) +
  geom_errorbar(aes(ymin = rating.q.25..gender, ymax = rating.q.75..gender, linetype = gender), 
                width = 0) +
  geom_errorbarh(aes(xmin = rating.q.25..age, xmax = rating.q.75..age, color = age), 
                 height = 0) +
  scale_x_continuous(name = 'Age (median)', 
                     breaks = c(min(df.gender_age$rating.median.age), max(df.gender_age$rating.median.age)),
                     labels = c('young', 'old')) + 
  scale_y_continuous(name = 'Gender (median)', 
                     breaks = c(min(df.gender_age$rating.median.gender), max(df.gender_age$rating.median.gender)),
                     labels = c('male', 'female')) + 
  facet_grid(. ~ type) 



gender_polarity_avg = ggplot(data = df.gender_polarity, 
                        aes(rating.mean.polarity, rating.mean.gender, color = polarity, fill = polarity, shape = gender, text = name)) +
  geom_point(size = 1.5) +
  geom_errorbar(aes(ymin = rating.mean.gender - rating.sd.gender, ymax = rating.mean.gender + rating.sd.gender, linetype = gender), 
                width = 0) +
  geom_errorbarh(aes(xmin = rating.mean.polarity - rating.sd.polarity, xmax = rating.mean.polarity + rating.sd.polarity, color = polarity), 
                 height = 0) +
  scale_x_continuous(name = 'Polarity (mean)', 
                     breaks = c(min(df.gender_polarity$rating.mean.polarity), max(df.gender_polarity$rating.mean.polarity)),
                     labels = c('evil', 'good')) + 
  scale_y_continuous(name = 'Gender (mean)', 
                     breaks = c(min(df.gender_polarity$rating.mean.gender), max(df.gender_polarity$rating.mean.gender)),
                     labels = c('male', 'female')) + 
  facet_grid(. ~ type) 

gender_polarity_med = ggplot(data = df.gender_polarity, 
                        aes(rating.median.polarity, rating.median.gender, color = polarity, fill = polarity, shape = gender, text = name)) +
  geom_point(size = 1.5) +
  geom_errorbar(aes(ymin = rating.q.25..gender, ymax = rating.q.75..gender, linetype = gender), 
                width = 0) +
  geom_errorbarh(aes(xmin = rating.q.25..polarity, xmax = rating.q.75..polarity, color = polarity), 
                 height = 0) +
  scale_x_continuous(name = 'Polarity (median)', 
                     breaks = c(min(df.gender_polarity$rating.median.polarity), max(df.gender_polarity$rating.median.polarity)),
                     labels = c('evil', 'good')) + 
  scale_y_continuous(name = 'Gender (median)', 
                     breaks = c(min(df.gender_polarity$rating.median.gender), max(df.gender_polarity$rating.median.gender)),
                     labels = c('male', 'female')) + 
  facet_grid(. ~ type) 
  

ggplotly(gender_polarity_avg, tooltip = "text")
ggplotly(gender_polarity_med, tooltip = "text")

ggplotly(gender_age_avg, tooltip = "text")
ggplotly(gender_age_med, tooltip = "text")


stat=df.gender_age %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(stat=cor.test(rating.mean.age,rating.mean.gender, method = 'pearson')[['statistic']])
corr=df.gender_age %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(corr=cor.test(rating.mean.age,rating.mean.gender, method = 'pearson')[['estimate']])
pval=df.gender_age %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(p=cor.test(rating.mean.age,rating.mean.gender, method = 'pearson')[['p.value']])
merge(corr, merge(stat, pval, by = 'type'), by = 'type')
#       type        corr       stat       p
#     madeup   -0.5080242  -3.538830  0.001129402
#       real   -0.2595825  -1.656973  0.105759986
#    talking   -0.3704647  -2.490779  0.017111613


stat=df.gender_age %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(stat=cor.test(rating.mean.age,rating.mean.gender, method = 'kendall')[['statistic']])
corr=df.gender_age %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(corr=cor.test(rating.mean.age,rating.mean.gender, method = 'kendall')[['estimate']])
pval=df.gender_age %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(p=cor.test(rating.mean.age,rating.mean.gender, method = 'kendall')[['p.value']])
merge(corr, merge(stat, pval, by = 'type'), by = 'type')
#    type       corr stat            p
#  madeup -0.3854908  216 0.0005209744
#    real -0.2025641  311 0.0671378802
# talking -0.2878049  292 0.0077257590


stat=df.gender_age %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(stat=cor.test(rating.median.age,rating.median.gender, method = 'pearson')[['statistic']])
corr=df.gender_age %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(corr=cor.test(rating.median.age,rating.median.gender, method = 'pearson')[['estimate']])
pval=df.gender_age %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(p=cor.test(rating.median.age,rating.median.gender, method = 'pearson')[['p.value']])
merge(corr, merge(stat, pval, by = 'type'), by = 'type')
# type       corr      stat            p
#    madeup -0.5160072 -3.614406 0.0009130574
#      real -0.2414083 -1.533496 0.1334381202
#   talking -0.3973741 -2.704280 0.0100922143


stat=df.gender_age %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(stat=cor.test(rating.median.age,rating.median.gender, method = 'kendall')[['statistic']])
corr=df.gender_age %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(corr=cor.test(rating.median.age,rating.median.gender, method = 'kendall')[['estimate']])
pval=df.gender_age %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(p=cor.test(rating.median.age,rating.median.gender, method = 'kendall')[['p.value']])
merge(corr, merge(stat, pval, by = 'type'), by = 'type')

#    type       corr      stat           p
#  madeup -0.3572457 -3.133696 0.001726198
#    real -0.2395910 -1.965417 0.049365943
# talking -0.2329653 -2.099935 0.035734519



stat=df.gender_polarity %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(stat=cor.test(rating.mean.polarity,rating.mean.gender, method = 'pearson')[['statistic']])
corr=df.gender_polarity %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(corr=cor.test(rating.mean.polarity,rating.mean.gender, method = 'pearson')[['estimate']])
pval=df.gender_polarity %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(p=cor.test(rating.mean.polarity,rating.mean.gender, method = 'pearson')[['p.value']])
merge(corr, merge(stat, pval, by = 'type'), by = 'type')
#    type      corr     stat           p
#  madeup 0.6110153 3.620323 0.001515383
#    real 0.5102459 2.446195 0.025610630
# talking 0.4764746 2.299292 0.033676363


stat=df.gender_polarity %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(stat=cor.test(rating.median.polarity,rating.median.gender, method = 'pearson')[['statistic']])
corr=df.gender_polarity %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(corr=cor.test(rating.median.polarity,rating.median.gender, method = 'pearson')[['estimate']])
pval=df.gender_polarity %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(p=cor.test(rating.median.polarity,rating.median.gender, method = 'pearson')[['p.value']])
merge(corr, merge(stat, pval, by = 'type'), by = 'type')
#    type      corr     stat            p
#  madeup 0.6325567 3.830730 0.0009105591
#    real 0.5044240 2.408685 0.0276364580
# talking 0.5106149 2.519576 0.0214174709


stat=df.gender_polarity %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(stat=cor.test(rating.mean.polarity,rating.mean.gender, method = 'kendall')[['statistic']])
corr=df.gender_polarity %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(corr=cor.test(rating.mean.polarity,rating.mean.gender, method = 'kendall')[['estimate']])
pval=df.gender_polarity %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(p=cor.test(rating.mean.polarity,rating.mean.gender, method = 'kendall')[['p.value']])
merge(corr, merge(stat, pval, by = 'type'), by = 'type')
#    type      corr stat           p
#  madeup 0.4420290  199 0.002081736
#    real 0.3567251  116 0.034430960
# talking 0.2526316  119 0.128413511


stat=df.gender_polarity %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(stat=cor.test(rating.median.polarity,rating.median.gender, method = 'kendall')[['statistic']])
corr=df.gender_polarity %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(corr=cor.test(rating.median.polarity,rating.median.gender, method = 'kendall')[['estimate']])
pval=df.gender_polarity %>%
  dplyr::group_by(type) %>%
  dplyr::summarize(p=cor.test(rating.median.polarity,rating.median.gender, method = 'kendall')[['p.value']])
merge(corr, merge(stat, pval, by = 'type'), by = 'type')
#    type      corr     stat           p
#  madeup 0.4770650 3.232161 0.001228578
#    real 0.4613435 2.449084 0.014322011
# talking 0.2826677 1.724061 0.084696773