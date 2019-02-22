# Application of Zeilis et al.'s algorithm to study stationarity patterns and growth
# Usage: R -f 02_zeilis_02_application.R
library(ggpubr)
library(strucchange)
se_ts <- read.csv("se_ts.csv")

# extracting breakdates and percentual growth between first and last period
datasets_levelbreaks_dates <- c()
datasets_per_break <- data.frame(break_nr=c(0), dataset_nr=c(0))
dates_of_first_break <- data.frame(month_nr=c(0), dataset_nr=c(0))
behavior_after_first_break <- data.frame(dataset_name=c(0), behavior=c(0))
dataset_names <- list()
datasets_with_no_breaks <- 0
datasets_levelbreaks_intervals <- c()
dataset_overall_growth_perc <- c()
se_ts_datasets <- as.vector(unique(se_ts$Name))
young_datasets <- c("3dprinting","ai","arabic","civicrm","coffee","computergraphics","crafts","economics","mythology","opensource","portuguese","elementaryos","emacs","engineering","es","esperanto","ethereum","moderators","monero","musicfans","startups","vi","woodworking","worldbuilding","retrocomputing","sitecore")  # datasets with less than 3years of history
nonyoung_datasets <- c()
for (dataset in se_ts_datasets) {
    dataset_ts <- ts(as.vector(se_ts[which(se_ts$Name == dataset), "Activity"]), frequency=1)
    dataset_levelbreaks <- breakpoints(dataset_ts ~ 1)
    dataset_levelbreaks_dates <- breakdates(dataset_levelbreaks)
    datasets_levelbreaks_dates <- c(datasets_levelbreaks_dates, dataset_levelbreaks_dates)
    datasets_levelbreaks_intervals <- c(datasets_levelbreaks_intervals, diff(dataset_levelbreaks_dates))
    number_breaks <- if (all(is.na(dataset_levelbreaks_dates))) 0 else length(dataset_levelbreaks_dates)
    month_of_first_break <- dataset_levelbreaks_dates[1]
    dataset_level_fit <- fitted(dataset_levelbreaks, breaks=number_breaks)
    if (number_breaks %in% datasets_per_break$break_nr) {
        row_index <- which(datasets_per_break$break_nr == number_breaks)
        datasets_per_break[[row_index, 2]] <- as.numeric(datasets_per_break[[row_index, 2]]) + 1
    } else {
        datasets_per_break <- rbind(datasets_per_break, c(number_breaks, 1))  # , dataset))
    }
    if (all(is.na(dataset_levelbreaks$breakpoints))) {
        #cat(dataset, "\n")
        datasets_with_no_breaks <- datasets_with_no_breaks + 1
    } else {
        if (month_of_first_break %in% dates_of_first_break$month_nr) {
            row_index <- which(dates_of_first_break$month_nr == month_of_first_break)
            dates_of_first_break[[row_index, 2]] <- dates_of_first_break[[row_index, 2]] + 1
            dataset_names[[month_of_first_break]] <- paste(dataset_names[[month_of_first_break]], dataset)
        } else {
            dates_of_first_break <- rbind(dates_of_first_break, c(month_of_first_break, 1))
            dataset_names[[month_of_first_break]] <- dataset
        }
    }
    if (!(dataset %in% young_datasets)) {
        dataset_levels <- unique(round(dataset_level_fit, 4))
        dataset_overall_growth_perc <- c(dataset_overall_growth_perc, dataset_levels[length(dataset_levels)] / dataset_levels[1] - 1)
        nonyoung_datasets <- c(nonyoung_datasets, dataset)
    }
}

# graphical analysis of breaks
dir.create("se_ts_breaks", showWarnings = F)
png("se_ts_breaks/how_many_datasets_Yaxis_have_a_break_in_Zweeks_xaxis.png")
plot(table(datasets_levelbreaks_dates), type="l")
dev.off()
png("se_ts_breaks/how_many_datasets_yaxis_have_xaxis_number_of_breaks.png")
ggbarplot(datasets_per_break, "break_nr", "dataset_nr", fill="#0072B2", color="#0072B2", label=T, lab.pos="in")
dev.off()
dates_of_first_break[1, 2] <- datasets_with_no_breaks  # putting here datasets with no break
png("se_ts_breaks/how_many_datasets_yaxis_have_month_xaxis_as_first_break_with_month0_equal_no_break.png")
ggbarplot(dates_of_first_break, "month_nr", "dataset_nr", fill="#0072B2", color="#0072B2", label=T, lab.pos="in")
dev.off()

# deriving growing and declining datasets
se_ts_quantiles <- quantile(dataset_overall_growth_perc, c(.2, .8))
(declining_datasets <- nonyoung_datasets[which(dataset_overall_growth_perc < se_ts_quantiles[1])])
# [1] "stackapps"        "webapps"          "sound"            "parenting"       
# [5] "ham"              "cooking"          "sustainability"   "pets"            
# [9] "spanish"          "tridion"          "boardgames"       "productivity"    
#[13] "pm"               "skeptics"         "expressionengine" "ebooks"          
#[17] "genealogy"        "craftcms"         "bricks"           "cstheory"        
#[21] "fitness"          "gardening"       
(growing_datasets <- nonyoung_datasets[which(dataset_overall_growth_perc > se_ts_quantiles[2])])
# [1] "tex"         "ru"          "electronics" "wordpress"   "dba"        
# [6] "codereview"  "puzzling"    "blender"     "salesforce"  "sharepoint" 
#[11] "crypto"      "askubuntu"   "gis"         "stats"       "security"   
#[16] "academia"    "opendata"    "ux"          "codegolf"    "money"      
#[21] "chemistry"   "unix"       
which(dataset_overall_growth_perc < se_ts_quantiles[1])
# [1]   1   3   4   6  16  30  40  43  44  45  57  58  61  62  63  70  73  75  78
#[20]  79  87 105
dataset_overall_growth_perc[which(dataset_overall_growth_perc < se_ts_quantiles[1])]
# [1] -0.8270164 -0.5000178 -0.3501128 -0.4013949 -0.5997053 -0.5081349
# [7] -0.7652142 -0.4066687 -0.5010268 -0.3606977 -0.2853363 -0.3510069
#[13] -0.6927358 -0.7308576 -0.6827223 -0.8151643 -0.8025591 -0.4124140
#[19] -0.6080018 -0.6163133 -0.3455895 -0.6110231
sort(dataset_overall_growth_perc[which(dataset_overall_growth_perc < se_ts_quantiles[1])])
# [1] -0.8270164 -0.8151643 -0.8025591 -0.7652142 -0.7308576 -0.6927358
# [7] -0.6827223 -0.6163133 -0.6110231 -0.6080018 -0.5997053 -0.5081349
#[13] -0.5010268 -0.5000178 -0.4124140 -0.4066687 -0.4013949 -0.3606977
#[19] -0.3510069 -0.3501128 -0.3455895 -0.2853363
sort(dataset_overall_growth_perc[which(dataset_overall_growth_perc > se_ts_quantiles[2])])
# [1] 1.692942 1.746882 1.822752 1.829010 1.942032 1.991754 1.997991 2.199795
# [9] 2.253730 2.300281 2.331064 2.407761 2.452632 2.961576 3.550105 4.039357
#[17] 4.066456 4.630583 4.734756 5.100631 7.364221 7.576156
