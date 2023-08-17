library(tidyverse)
library(latex2exp)
library(cowplot)

source("plot_utils.R")

############################################################
#                                                          #
#                        Load Data                         #
#                                                          #
############################################################

result = read_csv("../data/result_table.csv") %>% 
  mutate(dataset = ifelse(nchar(dataset)>3, str_to_title(dataset), str_to_upper(dataset))) %>% 
  mutate(dataset = ifelse(dataset == "Compas", "COMPAS", dataset))

result %>% 
  group_by(trainer, dataset) %>% 
  summarise(
    accs_sd = sd(accs,na.rm = T),
    mccs_sd = sd(mccs),
    uai_05_sd = sd(uai_05),
    uai_01_sd = sd(uai_01),
    uai_cf_sd = sd(uai_cf),
    uai_ar_05_sd = sd(uai_ar_05),
    uai_ar_01_sd = sd(uai_ar_01),
            accs = mean(accs),
            mccs = mean(mccs),
            uai_05 = mean(uai_05),
            uai_01 = mean(uai_01),
            uai_cf = mean(uai_cf),
            uai_ar_05 = mean(uai_ar_05),
            uai_ar_01 = mean(uai_ar_01)
            ) -> df_table

df_table %>% 
  select(trainer, dataset, accs, mccs, uai_05, uai_01, uai_cf, uai_ar_05, uai_ar_01) %>% 
  gather("indicator","mean",-trainer,-dataset) %>%
  filter(trainer!="SENSEI") %>% 
  mutate(isCapify = trainer == "CAPIFY") %>% 
  mutate(type = ifelse(dataset %in% c("Adult", "COMPAS"), "Real-world Data","Synthetic Data"))-> df1

df_table %>% 
  select(trainer, dataset, accs_sd, mccs_sd, uai_05_sd, uai_01_sd, uai_cf_sd, uai_ar_05_sd, uai_ar_01_sd) %>% 
  gather("indicator","std",-trainer,-dataset) %>%
  filter(trainer!="SENSEI") %>% 
  mutate( indicator= gsub("_sd","",indicator)) -> df2

df = merge(df1, df2, by = intersect(colnames(df1),colnames(df2)))

df_table %>% 
  filter(trainer!="SENSEI") %>%
  select(trainer, contains("_sd")) %>% 
  group_by(trainer) %>% 
  summarise(across(everything(), ~ mean(., na.rm = TRUE))) %>% 
  gather("indicator","std",-trainer) %>% 
  group_by(trainer) %>% 
  summarise(std = mean(std)) %>% 
  mutate(CAPIFY = ifelse(trainer == "CAPIFY", T, F )) %>% 
  group_by(CAPIFY) %>% 
  summarise(std = mean(std))
  

############################################################
#                                                          #
#                       Plots                              #
#                                                          #
############################################################
indicators = c("accs", "mccs", "uai_05", "uai_01", "uai_cf","uai_ar_05", "uai_ar_01")
ind_labels = c("Accuracy","MCC Score", 
               "Unfair Area ($\\Delta = 0.05$)",
               "Unfair Area ($\\Delta = 0.01$)", 
               "Counterfactual Unfair Area", 
               "Non-robust Area ($\\Delta = 0.05$)", 
               "Non-robust ($\\Delta = 0.01$)")

for(i in 1:7){
  p1 = df %>% 
    filter(indicator == indicators[i] ) %>% 
    filter(type == "Real-world Data") %>% 
    ggplot(aes(x = trainer, y = mean, fill = isCapify))+
    geom_col(position="dodge")+
    geom_errorbar(aes(ymin = mean - std, ymax = mean + std,color = isCapify), 
                  width = 0.2, position = position_dodge(0.9)) +
    facet_grid(type~dataset, scales = "free")+
    scale_y_continuous(labels = scales::percent)+
    labs(x = NULL, y = TeX(ind_labels[i]))+
    scale_fill_manual(values = bicols)+
    scale_color_manual(values = c("#0e305f","#a4521e"))+
    theme_Publication()+
    theme(
      axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
      axis.text.x = element_text(angle = 45,hjust = 1, size = 8,face = c('plain', 'plain','bold', 'plain', 'plain', 'plain')),
      legend.position = "none"
    )
  # p1
  p2 = df %>% 
    filter(indicator == indicators[i] ) %>% 
    filter(type != "Real-world Data") %>% 
    ggplot(aes(x = trainer, y = mean,fill = isCapify))+
    geom_col(position="dodge")+
    geom_errorbar(aes(ymin = mean - std, ymax = mean + std, color = isCapify), width = 0.2, 
                  position = position_dodge(0.9)) +
    facet_grid(type~dataset, scales = "free")+
    scale_y_continuous(labels = scales::percent)+
    labs(x = NULL, y = NULL)+
    scale_fill_manual(values = bicols)+
    scale_color_manual(values = c("#0e305f","#a4521e"))+
    theme_Publication()+
    theme(
      axis.text.x = element_text(angle = 45,hjust = 1, size = 8, face = c('plain', 'plain','bold', 'plain', 'plain', 'plain')),
      legend.position = "none"
    )
  p2
  p <- plot_grid(p1, p2, align = "h", rel_widths = c(1, 2))
  
  ggsave(paste0(indicators[i], ".pdf"),p,device = cairo_pdf,width = 40,height = 10,units = "cm")
  
}


############################################################
#                                                          #
#                         Table 1                          #
#                                                          #
############################################################
library(kableExtra)

df_table %>% 
  filter(trainer!="SENSEI") %>% 
  select(trainer, dataset,  accs, uai_05, uai_cf, uai_ar_05) %>% 
  ungroup()-> df1


pv1 <-  tidyr::pivot_wider(df1,names_from = dataset, values_from = c(accs,uai_05, uai_cf, uai_ar_05))

paste(1:ncol(pv1), colnames(tab1))
pv1 = pv1[,c(1, 2,8,14,20, 3,9,15,21, 4,10,16,22, 5,11,17,23, 6,12,18,24, 7,13,19,25)]
st1 = pv1

for(i in 2:ncol(pv1)){
  pv1[,i] = as.character(round(unlist(pv1[,i]), 2))
  if(str_detect(colnames(pv1)[i],"accs|mcc")){
    i_max = which.max(unlist(st1[,i]))  
  } else{
    i_max = which.min(unlist(st1[,i]))
  }
  
  
  pv1[i_max, i] = sprintf("textbf(%s)",pv1[i_max,i])
}
kbl(pv1, booktabs = T,format = "latex")





############################################################
#                                                          #
#                         Table 2                          #
#                                                          #
############################################################

df_table %>% 
  filter(trainer!="SENSEI") %>% 
  select(trainer, dataset,  mccs, uai_01, uai_ar_01) %>% 
  ungroup()-> df1


pv1 <-  tidyr::pivot_wider(df1,names_from = dataset, values_from = c(mccs, uai_01, uai_ar_01))

paste(1:ncol(pv1), colnames(pv1))
pv1 = pv1[,c(1, 2,8,14, 3,9,15, 4,10,16, 5,11,17, 6,12,18, 7,13,19)]
st1 = pv1

for(i in 2:ncol(pv1)){
  pv1[,i] = as.character(round(unlist(pv1[,i]), 2))
  if(str_detect(colnames(pv1)[i],"accs|mcc")){
    i_max = which.max(unlist(st1[,i]))  
  } else{
    i_max = which.min(unlist(st1[,i]))
  }
  
  
  pv1[i_max, i] = sprintf("textbf(%s)",pv1[i_max,i])
}
kbl(pv1, booktabs = T,format = "latex")




df_table %>% 
  filter(trainer!="SENSEI") %>% 
  select(trainer, dataset,  mccs,uai_cf) -> df1


pivot_wider(df1, names_from = dataset, 
            values_from = c("mccs", "uai_cf")) -> tab1

tab1 = tab1[,c(1, 2,8, 3,9, 4,10, 5,11, 6,12, 7,13)]

st1 = tab1

for(i in 2:ncol(tab1)){
  tab1[,i] = as.character(round(unlist(tab1[,i]), 2))
  if(str_detect(colnames(tab1)[i],"accs")){
    i_max = which.max(unlist(st1[,i]))  
  } else{
    i_max = which.min(unlist(st1[,i]))
  }
  
  
  tab1[i_max, i] = sprintf("textbf(%s)",tab1[i_max,i])
}
kbl(tab1, booktabs = T,format = "latex")


############################################################
#                                                          #
#                         MCC Plot                         #
#                                                          #
############################################################


i  = 2
p1 = df %>% 
  filter(indicator == indicators[i] ) %>% 
  filter(type == "Real-world Data") %>% 
  ggplot(aes(x = trainer, y = mean, fill = isCapify))+
  geom_col(position="dodge")+
  geom_errorbar(aes(ymin = mean - std, ymax = mean + std,color = isCapify), 
                width = 0.2, position = position_dodge(0.9)) +
  facet_grid(type~dataset, scales = "free")+
  #scale_y_continuous(labels = scales::percent)+
  labs(x = NULL, y = TeX(ind_labels[i]))+
  scale_fill_manual(values = bicols)+
  scale_color_manual(values = c("#0e305f","#a4521e"))+
  theme_Publication()+
  theme(
    axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
    axis.text.x = element_text(angle = 45, size = 8, hjust = 1,face = c('plain', 'plain','bold', 'plain', 'plain', 'plain')),
    legend.position = "none"
  )
p2 = df %>% 
  filter(indicator == indicators[i] ) %>% 
  filter(type != "Real-world Data") %>% 
  ggplot(aes(x = trainer, y = mean,fill = isCapify))+
  geom_col(position="dodge")+
  geom_errorbar(aes(ymin = mean - std, ymax = mean + std, color = isCapify), width = 0.2, 
                position = position_dodge(0.9)) +
  facet_grid(type~dataset, scales = "free")+
  #scale_y_continuous(labels = scales::percent)+
  labs(x = NULL, y = NULL)+
  scale_fill_manual(values = bicols)+
  scale_color_manual(values = c("#0e305f","#a4521e"))+
  theme_Publication()+
  theme(
    axis.text.x = element_text(angle = 45, size = 8, hjust = 1,face = c('plain', 'plain','bold', 'plain', 'plain', 'plain')),
    legend.position = "none"
  )
p2
p <- plot_grid(p1, p2, align = "h", rel_widths = c(1, 2))

ggsave(paste0(indicators[i], ".pdf"),p,device = cairo_pdf,width = 40,height = 10,units = "cm")
