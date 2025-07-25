pkgs <- c("ggplot2","patchwork","readxl","UpSetR","data.table")
for (pkg in pkgs){
  suppressPackageStartupMessages(library(pkg, character.only = T))
}
#####################################################################
############################# Fig. 3c ###############################
#####################################################################
p = ggplot(lambda, aes(lambda, accuracy, group=1, color="#036eb8"))+
  geom_line(linewidth=1)+
  geom_point(size=3)+
  scale_color_manual(values = c("#036eb8"))+
  scale_y_continuous(limits = c(0.63, 0.76))+
  scale_x_discrete(labels=c("1e-02","5e-03","1e-03","5e-04","1e-04","5e-05"))+
  ylab("Accuracy")+
  xlab("Lambda")+
  theme_bw()+
  theme(axis.text = element_text(size=13, color = "black"),
        axis.title = element_text(size=14),
        panel.grid = element_blank(),
        legend.position = "none")


#####################################################################
############################# Fig. 3d ###############################
#####################################################################
p = ggplot(metric, aes(x=Method,color=Method,fill=Method))+
  geom_bar(aes(y=Accuracy), stat="identity", width = 0.6)+
  scale_color_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  scale_fill_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  scale_y_continuous(limits = c(0,0.84))+
  theme_bw()+
  theme(axis.text = element_text(size=13, color = "black"),
        axis.title = element_text(size=14),
        panel.grid = element_blank())


#####################################################################
############################# Fig. 3e ###############################
#####################################################################
p = ggplot(data, aes(FPR, TPR, group=method, color=method))+
  geom_line(linewidth=1)+
  scale_color_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  theme_bw()+
  theme(axis.text = element_text(size=13, color = "black"),
        axis.title = element_text(size=14),
        panel.grid = element_blank())


#####################################################################
############################# Fig. 3f ###############################
#####################################################################
load("selected_features.Rdata")
upset(fromList(selected_features),
      text.scale=c(2,2,2,2,2,2))


#####################################################################
############################# Fig. 3g ###############################
#####################################################################
file_path <- "features_literature.xlsx"
sheet_names <- excel_sheets(file_path)
sheet_data <- lapply(sheet_names[1:5], function(sheet) {
  read_excel(file_path, sheet = sheet)
})

df <- c()
for(i in 1:5){
  feature = sheet_data[[i]][[1]]
  phy_feature <- feature[grep("_pt_", feature)]
  if(i<4){
    df_tmp = data.frame(method=sheet_names[i],num=c(length(phy_feature),length(feature)-length(phy_feature),0),type=c("Phylotype","Taxonomy","reported"))
  }else{
    report = na.omit(sheet_data[[i]][[2]])
    df_tmp = data.frame(method=sheet_names[i],num=c(length(phy_feature),length(feature)-length(phy_feature),length(report)),type=c("Phylotype","Taxonomy","reported"))
  }
  df <- rbind(df, df_tmp)
}
df$type <- factor(df$type, levels = c("Phylotype","Taxonomy","reported"))
df$method <- factor(df$method, levels = c("DeepVaris","SurvNet","DeepLINK","HiDe-MK","Stabl"))
p = ggplot(df, aes(x = method, y = num, group = type, color=type, fill=type)) +
  geom_bar(stat = "identity", position = position_dodge2(padding = 0.2), alpha=0.5) +
  scale_color_manual(values = c("#ffa510","#036eb8","#036eb8"))+
  scale_fill_manual(values = c("white","white","#036eb8"))+
  scale_y_continuous(limits = c(0,155))+
  theme_bw()+
  ylab("No. of features")+
  xlab("")+
  theme(axis.text = element_text(size=13, color = "black"),
        axis.title = element_text(size=14),
        panel.grid = element_blank())

