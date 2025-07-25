pkgs <- c("ggplot2","patchwork","readxl","ggbreak")
for (pkg in pkgs){
  suppressPackageStartupMessages(library(pkg, character.only = T))
}
#####################################################################
############################# Fig. 2b ###############################
#####################################################################
p = ggplot(res, aes(x=method,color=method,fill=method))+
  geom_errorbar(aes(x=method, ymin=pmax(0,Select_Feature_Num_mean-Select_Feature_Num_std), ymax=Select_Feature_Num_mean+Select_Feature_Num_std), stat="identity", width=0.5, position=position_dodge(0.7))+
  geom_bar(aes(y=Select_Feature_Num_mean), stat="identity", position=position_dodge(), width = 0.7)+
  scale_y_continuous(limits = c(0,400),
                     breaks = c(0,64,100))+
  scale_y_break(c(100,370), scales="fixed",
                ticklabels=c(370,400),
                expand=expansion(add = c(0,0)),
                space=0.1)+
  scale_color_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  scale_fill_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  theme_bw()+
  ylab("No. of features")+
  facet_wrap(facets = "Sheet", ncol = 6)+
  theme(axis.text = element_text(size=11, color = "black"),
        axis.title = element_text(size=12),
        panel.grid = element_blank(),
        legend.position = "none")


#####################################################################
############################# Fig. 2c ###############################
#####################################################################
p <- list()
for(i in 1:6){
  res_sheet <- res[res$Sheet==paste0("dataset",i),] 
  p[[i]] <- ggplot(res_sheet, aes(x = method, color=method, fill=method)) +
    geom_point(aes(y = Final_Error_mean), size = 4) +
    geom_errorbar(aes(x=method, ymin=pmax(0,Final_Error_mean-Final_Error_std), ymax=Final_Error_mean+Final_Error_std), stat="identity", width=0.2, position=position_dodge())+
    scale_color_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
    scale_fill_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
    labs(y = "Final error", x = "") +
    theme_bw()+
    theme(axis.text = element_text(size=11, color = "black"),
          axis.title = element_text(size=12),
          plot.title = element_text(hjust = 0.5, size = 12),
          panel.grid = element_blank(),
          legend.position = "none")+
    ggtitle(paste0("Dataset ",i)) 
}


#####################################################################
############################# Fig. 2d ###############################
#####################################################################
p <- ggplot(tr_all, aes(method,TR,color=method,fill=method))+
  stat_boxplot(geom = "errorbar")+
  geom_boxplot(outliers = FALSE)+
  scale_color_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  scale_fill_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  scale_y_continuous(limits = c(0,1))+
  facet_wrap(facets = "data", ncol = 6)+
  theme_bw()+
  theme(panel.grid=element_blank(),
        legend.position = "right",
        axis.text = element_text(size=11, color = "black"),
        axis.title = element_text(size=12))


#####################################################################
############################# Fig. 2e ###############################
#####################################################################
p <- ggplot(fdr_all, aes(method,FDR,color=method,fill=method))+
  stat_boxplot(geom = "errorbar")+
  geom_boxplot(outliers = FALSE)+
  scale_color_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  scale_fill_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  scale_y_continuous(limits = c(0,1))+
  facet_wrap(facets = "data", ncol = 6)+
  theme_bw()+
  theme(panel.grid=element_blank(),
        legend.position = "right",
        axis.text = element_text(size=11, color = "black"),
        axis.title = element_text(size=12))


