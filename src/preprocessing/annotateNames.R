library(dplyr)

weird_cases = c("Dal", "Tansy", "Nova", "Thresh", "Fortuna", "Hannerl", "Serafina", "Saiorse", 
                "Alasdair", "Natasha", "Nessa", "April", "Bonny", "Flora", "Hope", "Myrtle", 
                "Pearl", "Sunny", "Violet")

real = c("Adelaide", "Alasdair", "Alice", "Allie", "Chloe", "Claire", "Derrick", "Enid", "Ethel", "Fiorella", 
         "Francis", "Freddie", "Fred", "George", "Gerald", "Gerard", "Harold", "Henry", "Ian", "Jackie", 
         "Joseph", "Karl", "Kathryn", "Marion", "Maud", "Mildred", "Mina", "Molly", "Natasha", "Nina", "Ralph", 
         "Rosalind", "Simon", "Stanley", "Stuart", "Susan", "Sybill", "Tristram", "William", "Adelina", "Beatrice", 
         "Caroline", "Damon", "Elena", "Jude", "Julia", "Klaus", "Lee", "Lucy", "Lyra", "Marisa", "Matthias", 
         "Nessa", "Peter", "Tobias", "Will", "Millicent", "Ginny", "Jadis", "Ma", "Virginia")

talking = c("April", "Ball", "Blackberry", "Bonny", "Brilliantine", "Cackle", "Coot", "Dal", "Dark", "Day", 
            "Desire", "Ditto", "Dove", "Drill", "Flora", "Fox", "Ground", "Guy", "Hope", "Joy", "Just", "Kitty", 
            "Lavender", "Silky", "Myrtle", "Mouse", "Parrish", "Pearl", "Sprout", "Poop", "Roach", "Scullery", 
            "Skinner", "Sparrow", "Tansy", "Travers", "Winky", "Wolf", "Wren", "Yak", "Apple", "Berry", "Blue", 
            "Clove", "Diamond", "Flick", "Gale", "Glimmer", "Goth", "Jewel", "Lark", "Marvel", "Nova", "Scotch", 
            "Spark", "Sunny", "Sway", "Thresh", "Touchstone", "Steepy", "Violet")

madeup = c("Alastor", "Alecto", "Araminta", "Argus", "Asim", "Chi", "Cooki", "Dalip", "Dumpa", "Farder", "Fenrir", 
           "Fortuna", "Griphook", "Grunter", "Hannerl", "Hepzibah", "Ione", "Iorek", "Jungli", "Kreacher", "Lak", 
           "Luft", "Minna", "Miskouri", "Muffet", "Mundungus", "Nergui", "Nitasha", "Robbo", "Skellig", "Steg", 
           "Talentino", "Tasha", "Titania", "Yozadah", "Zahara", "Zubaida", "Amabala", "Arcturus", "Arobynn", "Brum", 
           "Cardan", "Chaol", "Elide", "Gmork", "Goha", "Kaisa", "Kaz", "Levana", "Lorcan", "Mogget", "Morgra", 
           "Neoma", "Penthe", "Serafina", "Tenar", "Saiorse", "Schaffa", "Inej", "Spink", "Lorcan", "Losh",
           "Mogget")


setwd("/Volumes/University/TiU/Research/Research Traineeship/2020_21-characters/")

ff_names = read.csv("names_LDA/data/fan-fiction/orthoFeatures_polarity.csv")[,1:3]
ya_names = read.csv("names_LDA/data/Antwerp/cade.csv")[,1:3]
names = merge(ya_names, ff_names, by=c("name", "gender"), all = T)
names = names %>%
  mutate(type = case_when(
    name %in% madeup ~ "made-up",
    name %in% real ~ "real",
    name %in% talking ~ "talking"))

names = names %>%
  mutate(flag = if_else(name %in% weird_cases, 1, 0))

write.csv(names, "Data/names_annotated.csv", row.names = F, na = '')
