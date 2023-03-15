library(ggplot2)
library(lme4)
library(lmerTest)
library(ggeffects)
library(MASS)
library(plyr)
library(dplyr)
library(tidyr)
library(moments)
library(mgcv)
library(itsadug)


# semantic differentials:
# negative values                 positive values
# male                            female
# evil                            good
# young                           old

setwd('/Volumes/University/TiU/Research/ResearchTraineeship/2020_21-characters/LangCogn/')


df.item = read.csv('data/Ratings_annotated.csv')
df.item$name = factor(df.item$name)
df.item$age = factor(df.item$age)
df.item$polarity = factor(df.item$polarity)
df.item$Prolific_ID = factor(df.item$Prolific_ID)
df.item$attribute = factor(df.item$attribute)
df.item$type = factor(df.item$type)
df.item$gender = factor(df.item$gender)


##### descriptives #####
ggplot(data = df.item[df.item$attribute == 'age',], aes(x=rating)) +
  geom_density(aes(group = name, color = age, linetype = age)) +
  scale_color_manual(values = c(
    "young" = "darkorchid1",
    "old" = "darkorchid4"
  )) +
  facet_grid(. ~ type) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right'
  )


ggplot(data = df.item[df.item$attribute == 'polarity' & df.item$polarity != 'ambiguous',], aes(x=rating)) +
  geom_density(aes(group = name, color = polarity, linetype = polarity)) +
  scale_color_manual(values = c(
    "good" = "steelblue1",
    "evil" = "steelblue4"
  )) +
  facet_grid(. ~ type) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right'
  )


ggplot(data = df.item[df.item$attribute == 'gender',], aes(x = rating)) +
  geom_density(aes(group = name, color = gender, linetype = gender)) +
  xlim(-49,49) +
  scale_color_manual(values = c(
    "female" = "darkgoldenrod4",
    "male" = "darkgoldenrod1"
  )) +
  facet_grid(. ~ type) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right'
  )



skew=df.item %>%
  dplyr::group_by(name, type, attribute) %>%
  dplyr::summarize(k=skewness(rating))

ggplot(data = skew, aes(x = abs(k))) +
  geom_density(aes(color = attribute, linetype = attribute), linewidth=2) +
  scale_color_manual(values = c(
    "polarity" = "steelblue1",
    "gender" = "darkgoldenrod1",
    "age" = "darkorchid1"
  )) +
  facet_grid(. ~ type) +
  labs(x='skeweness') +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right'
  )


df.aggr = read.csv('data/avgRatings_annotated.csv', header = T)
df.aggr.pol = df.aggr %>%
  filter(polarity %in% c("evil", "good")) %>%
  select(name, type, gender, polarity, attribute, rating.mean, rating.sd) %>%
  pivot_wider(names_from = attribute, values_from = c(rating.mean, rating.sd))

df.aggr.age = df.aggr %>%
  filter(age %in% c("young", "old")) %>%
  select(name, type, gender, age, attribute, rating.mean, rating.sd) %>%
  pivot_wider(names_from = attribute, values_from = c(rating.mean, rating.sd))

ggplot(data = df.aggr.pol, aes(x=rating.mean_gender, y=rating.mean_polarity)) +
  geom_point(aes(color = gender, shape = polarity), size = 2) +
  geom_errorbar(aes(ymax = rating.mean_polarity + rating.sd_polarity, ymin = rating.mean_polarity - rating.sd_polarity, color = gender, linetype = polarity), width = 1) +
  geom_errorbarh(aes(xmax = rating.mean_gender + rating.sd_gender, xmin = rating.mean_gender - rating.sd_gender, color = gender, linetype = polarity), height = 1) +
  scale_color_manual(values = c(
    "female" = "darkgoldenrod4",
    "male" = "darkgoldenrod1"
  )) +
  labs(x='gender rating', y='polarity rating') +
  scale_y_continuous(breaks=c(-50, 0, 50), labels = c('evil', ' ', 'good')) +
  scale_x_continuous(breaks=c(-50, 0, 50), labels = c('male', ' ', 'female')) +
  facet_grid(. ~ type) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'horizontal', 
    legend.box.just = 'right'
  )

ggplot(data = df.aggr.age, aes(x=rating.mean_gender, y=rating.mean_age)) +
  geom_point(aes(color = gender, shape = age), size = 2) +
  geom_errorbar(aes(ymax = rating.mean_age + rating.sd_age, ymin = rating.mean_age - rating.sd_age, color = gender, linetype = age), width = 1) +
  geom_errorbarh(aes(xmax = rating.mean_gender + rating.sd_gender, xmin = rating.mean_gender - rating.sd_gender, color = gender, linetype = age), height = 1) +
  scale_color_manual(values = c(
    "female" = "darkgoldenrod4",
    "male" = "darkgoldenrod1"
  )) +
  labs(x='gender rating', y='age rating') +
  scale_y_continuous(breaks=c(-50, 0, 50), labels = c('young', ' ', 'old')) +
  scale_x_continuous(breaks=c(-50, 0, 50), labels = c('male', ' ', 'female')) +
  facet_grid(. ~ type) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'horizontal', 
    legend.box.just = 'right'
  )

##### reality from ratings #####

## age
age.df = df.item[df.item$attribute == 'age', ]
age.df$age = relevel(age.df$age, 'young')
levels(age.df$age)
age.lm.base = glm(age ~ type, data = age.df, family = binomial(link = "logit"))
summary(age.lm.base)
AIC(age.lm.base)  # 5516.142

age.lm.rating = glm(age ~ type + rating, data = age.df, family = binomial(link = "logit"))
summary(age.lm.rating)
AIC(age.lm.rating)  # 5490.984
AIC(age.lm.base) - AIC(age.lm.rating)  # 25.15802

age.lm.rating_int = glm(age ~ type*rating, data = age.df, family = binomial(link = "logit"))
summary(age.lm.rating_int)
AIC(age.lm.rating_int) # 5300.881
AIC(age.lm.base) - AIC(age.lm.rating_int)  # 215.2611
AIC(age.lm.rating) - AIC(age.lm.rating_int)  # 190.103

age.gam.rating_int = gam(age ~ s(rating) + s(rating, by=type), data = age.df, family = binomial(link = "logit"))
AIC(age.gam.rating_int)  # 5279.924
plot_smooth(
  age.gam.rating_int, view='rating', plot_all = 'type', rug=TRUE, rm.ranef=TRUE,
  main = "ratings predicting ageder by name type", 
  xlab = 'rating', ylab = 'p(old)', lty = 1
)

age.preds = data.frame(
  ggpredict(age.lm.rating_int, terms = c(
    "rating[all]",
    "type[all]")
  )
)
names(age.preds)[names(age.preds) == 'x'] <- 'rating'
names(age.preds)[names(age.preds) == 'group'] <- 'NameType'
names(age.preds)[names(age.preds) == 'predicted'] <- 'prob'
age.preds$attribute = 'age'

ggplot(data = age.preds, aes(x = rating, y = prob)) +
  geom_line(aes(color = NameType)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = NameType, fill=NameType), alpha = 0.4) +
  ggtitle('Age')

# ratings on the semantic differential young - old predict whether the name was originally used for
# and old or young character, but there is a significant interaction between rating and name type:
# - the probability that a character was indeed old increases when ratings are higher (towards old) 
#     for made-up names
# - the probability that a character was indeed old isn't related to the ratings for talking names
# - the probability that a character was indeed old decreases when ratings are higher (towards old)
#     for real names, suggesting that our raters disagree with the authors.


## polarity
pol.df = df.item[df.item$attribute == 'polarity' & df.item$polarity != 'ambiguous', ]
pol.df = droplevels(pol.df)
pol.df$polarity = relevel(pol.df$polarity, 'evil')
levels(pol.df$polarity)
pol.lm.base = glm(polarity ~ type, data = pol.df, family = binomial(link = "logit"))
summary(pol.lm.base)
AIC(pol.lm.base)  # 2383.824

pol.lm.rating = glm(polarity ~ type + rating, data = pol.df, family = binomial(link = "logit"))
summary(pol.lm.rating)
AIC(pol.lm.rating)  # 2274.323
AIC(pol.lm.base) - AIC(pol.lm.rating)  # 109.5011

pol.lm.rating_int = glm(polarity ~ type*rating, data = pol.df, family = binomial(link = "logit"))
summary(pol.lm.rating_int)
AIC(pol.lm.rating_int) # 2275.005
AIC(pol.lm.base) - AIC(pol.lm.rating_int)  # 108.8187
AIC(pol.lm.rating) - AIC(pol.lm.rating_int)  # -0.6823831

pol.gam.rating_int = gam(polarity ~ s(rating) + s(rating, by=type), data = pol.df, family = binomial(link = "logit"))
AIC(pol.gam.rating_int)  # 2311.277
plot_smooth(
  pol.gam.rating_int, view='rating', plot_all = 'type', rug=TRUE, rm.ranef=TRUE,
  main = "ratings predicting polder by name type", 
  xlab = 'rating', ylab = 'p(good)', lty = 1
)

pol.preds = data.frame(
  ggpredict(pol.lm.rating, terms = c(
    "rating[all]",
    "type[all]")
  )
)
names(pol.preds)[names(pol.preds) == 'x'] <- 'rating'
names(pol.preds)[names(pol.preds) == 'group'] <- 'NameType'
names(pol.preds)[names(pol.preds) == 'predicted'] <- 'prob'
pol.preds$attribute = 'polarity'

ggplot(data = pol.preds, aes(x = rating, y = prob)) +
  geom_line(aes(color = NameType)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = NameType, fill=NameType), alpha = 0.4) +
  ggtitle('Polarity')

# no significant interaction between ratings on the evil - good semantic differential and the name
# type. The effect of rating on the probability of a character being actually good is significant 
# and positive: ratings indicating a name better fitted a good character predict a character is 
# likelier to be actually good in the original fan fiction. This effect is the same for real, made-up,
# and talking names.


## gender
gen.df = df.item[df.item$attribute == 'gender', ]
gen.df$gender = relevel(gen.df$gender, 'male')
levels(gen.df$gender)
gen.lm.base = glm(gender ~ type, data = gen.df, family = binomial(link = "logit"))
summary(gen.lm.base)
AIC(gen.lm.base)  # 8845.738

gen.lm.rating = glm(gender ~ type + rating, data = gen.df, family = binomial(link = "logit"))
summary(gen.lm.rating)
AIC(gen.lm.rating)  # 5959.238
AIC(gen.lm.base) - AIC(gen.lm.rating)  # 2886.5

gen.lm.rating_int = glm(gender ~ type*rating, data = gen.df, family = binomial(link = "logit"))
summary(gen.lm.rating_int)
AIC(gen.lm.base) - AIC(gen.lm.rating_int)  # 3218.315
AIC(gen.lm.rating) - AIC(gen.lm.rating_int)  # 331.8147

gen.gam.rating_int = gam(gender ~ s(rating) + s(rating, by=type), data = gen.df, family = binomial(link = "logit"))
AIC(gen.gam.rating_int)  # 5571.139
plot_smooth(
  gen.gam.rating_int, view='rating', plot_all = 'type', rug=TRUE, rm.ranef=TRUE,
  main = "ratings predicting gender by name type", 
  xlab = 'rating', ylab = 'p(female)', lty = 1
)

gen.preds = data.frame(
  ggpredict(gen.lm.rating_int, terms = c(
    "rating[all]",
    "type[all]")
  )
)
names(gen.preds)[names(gen.preds) == 'x'] <- 'rating'
names(gen.preds)[names(gen.preds) == 'group'] <- 'NameType'
names(gen.preds)[names(gen.preds) == 'predicted'] <- 'prob'
gen.preds$attribute = 'gender'

ggplot(data = gen.preds, aes(x = rating, y = prob)) +
  geom_line(aes(color = NameType)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = NameType, fill=NameType), alpha = 0.4)  +
  ggtitle("Gender")

# There is a significant interaction between name type and ratings on the gender semantic differential
# in predicting the probability that a name referred to a female character. Even though the effect is 
# positive regardless of name type, with ratings closer to the female pole predicting names referring
# to female characters, the effect is stronger for real names (as expected). 


preds = rbind(gen.preds, age.preds, pol.preds)

ggplot(data = preds, aes(x = rating, y = prob)) +
  geom_line(aes(color = NameType)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = NameType, fill=NameType), alpha = 0.4)  +
  facet_grid(. ~ NameType) +
  scale_color_manual(values = c(
    "madeup" = "turquoise4",
    "real" = "firebrick3",
    "talking" = "darkorange1"
  )) +
  scale_fill_manual(values = c(
    "madeup" = "turquoise4",
    "real" = "firebrick3",
    "talking" = "darkorange1"
  )) +
  facet_grid(. ~ attribute) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right'
  ) +
  ggtitle("Relation between ratings and real attributes")
