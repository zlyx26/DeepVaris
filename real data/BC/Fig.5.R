pkgs <- c("ggplot2","patchwork","readxl","UpSetR","data.table","Seurat","dplyr",
          "Rtsne","uwot","cluster","tibble","ggrepel","org.Hs.eg.db","tidyr",
          "clusterProfiler","ReactomePA","stringr","ggsankeyfier","RColorBrewer","slingshot")
for (pkg in pkgs){
  suppressPackageStartupMessages(library(pkg, character.only = T))
}
#####################################################################
############################# Fig. 5b ###############################
#####################################################################
dir = c("./TNBC/","./ER+/","./HER2+/")
for(d in dir){
  metric = fread(paste0(d,"accuracy_auc_results.csv"))
  colnames(metric) = c("Method","Accuracy","AUC")
  metric$Method <- factor(metric$Method, levels = c("DeepVaris","Survnet","Deep LINK","HiDe","Stabl"))
  
  p = ggplot(metric, aes(x=Method,color=Method,fill=Method))+
    geom_bar(aes(y=Accuracy), stat="identity", width = 0.8)+
    scale_color_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
    scale_fill_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
    scale_y_continuous(limits = c(0,max(metric$Accuracy+0.05)))+
    theme_bw()+
    theme(axis.text = element_text(size=13, color = "black"),
          axis.title = element_text(size=14),
          panel.grid = element_blank())
}


#####################################################################
############################# Fig. 5c ###############################
#####################################################################
dir = c("./TNBC/","./ER+/","./HER2+/")
for(d in dir){
  method = c("deepvaris","survnet","DeepLINK","HiDe","stabl")
  data = c()
  for(met in method){
    roc <- fread(paste0(d, met, "_roc.csv"))
    roc$method = met
    data = rbind(data, roc)
  }
  data$method <- factor(data$method, levels = method)
  p = ggplot(data, aes(FPR, TPR, group=method, color=method))+
    geom_line(linewidth=1)+
    scale_color_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
    theme_bw()+
    theme(axis.text = element_text(size=13, color = "black"),
          axis.title = element_text(size=14),
          panel.grid = element_blank())
}


#####################################################################
############################# Fig. 5d ###############################
#####################################################################
load("selected_features.Rdata")
for(d in c("TNBC","ER","HER2")){
  features = as.data.frame(selected_features[[d]])
  res <- list()
  for(i in 1:5){
    feature = na.omit(features[,i])
    res[[colnames(features)[i]]] <- feature
  }
  upset(fromList(res),text.scale=c(2,2,2,2,2,2))
}


#####################################################################
############################# Fig. 5e ###############################
#####################################################################
for(j in 1:3){
  file_path <- paste0(dir[j],"features_liter.xlsx")
  sheet_names <- excel_sheets(file_path)
  
  sheets_data <- lapply(sheet_names, function(sheet) {
    read_excel(file_path, sheet = sheet)
  })
  marker_num = c()
  for(i in 1:length(sheets_data)){
    dat = as.data.frame(sheets_data[[i]])
    num = sum(is.na(dat$reported))
    marker_df = data.frame(type=c("other","reported"),num=c(num,nrow(dat)-num),method=sheet_names[i])
    marker_num = rbind(marker_num,marker_df)
  }
  marker_num$type = factor(marker_num$type, levels = c("other","reported"))
  marker_num$method = factor(marker_num$method, levels = sheet_names)
  p1 = ggplot(marker_num, aes(x=method,y=num,fill=type,color=method))+
    geom_bar(stat="identity", position="stack", width = 0.6, linewidth = 1)+
    scale_fill_manual(values = c("white","#eca680"))+
    scale_color_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
    theme_bw()+
    theme(axis.text = element_text(size=13, color = "black"),
          axis.title = element_text(size=14),
          panel.grid = element_blank())
}


#####################################################################
############################# Fig. 5f ###############################
#####################################################################
features = as.data.frame(selected_features[["TNBC"]])
dv_only = setdiff(na.omit(features[,1]),na.omit(features[,2]))
for(j in 3:ncol(features)){
  dv_only = setdiff(dv_only,na.omit(features[,j]))
}
g_co <- c()
marker = data.frame(gene=dv_only)
colnames(marker) = "gene"
ids = bitr(marker$gene,'SYMBOL','ENTREZID',"org.Hs.eg.db") 
cutoff <- 0.05

m1 <- merge(marker,ids,by.x='gene',by.y='SYMBOL')
go1up <- enrichGO(gene = m1$ENTREZID,  
                  OrgDb = org.Hs.eg.db, 
                  keyType = 'ENTREZID',  
                  ont = 'ALL',  
                  pAdjustMethod = 'fdr',  
                  pvalueCutoff = cutoff,  
                  qvalueCutoff = 0.2)  

df1up <- c()
if(is.null(go1up)==FALSE) {
  df1up <- go1up@result
  df1up <- df1up[order(df1up$pvalue,decreasing = F),]
  df1up <- subset(df1up,df1up$pvalue<cutoff)
  df1up$Log <- -log10(df1up$pvalue)
}
g_co <- rbind(g_co, df1up)

df1 <- g_co[grep("T cell", g_co$Description), ]
df2 <- g_co[grep("lymphocyte", g_co$Description), ]
df = rbind(df1,df2)
df$enrichment_factor <- 0
for(i in 1:nrow(df)){    
  df[i,]$enrichment_factor <- eval(parse(text = df[i,]$GeneRatio))/eval(parse(text = df[i,]$BgRatio))
}
df <- df[order(df$Log),]
df <- df %>% distinct(Description, .keep_all = TRUE)
df$Description <- factor(df$Description, levels = df$Description)

df_sep = df %>%
  separate_rows(.,geneID,convert=TRUE,sep="/")
ids = bitr(df_sep$geneID,'ENTREZID','SYMBOL',"org.Hs.eg.db") 
colnames(ids) = c("geneID","geneName")
df_sep$geneID = as.character(df_sep$geneID)
df_sep = left_join(df_sep,ids,by="geneID")

dff <- df_sep %>% select(geneName,Description,Count) %>%
  pivot_stages_longer(
    stages_from=c("geneName","Description"),
    values_from ="Count")

p1 = ggplot(data=dff,aes(x=stage,y=Count,
                         group = node,
                         edge_id = edge_id,
                         connector = connector)) +
  geom_sankeyedge(
    position=position_sankey(order="ascending",
                             v_space ="auto",
                             width = 0.05)) +
  geom_sankeynode(aes(fill=node,color=node),
                  position=position_sankey(
                    order="ascending",v_space="auto",width=0.05)) +
  geom_text(data=dff %>% filter(connector=="from"),
            aes(label=node),stat="sankeynode",
            position=position_sankey(
              v_space="auto",order="ascending",nudge_x=-0.05),
            hjust=1,size=5,color="black") +
  geom_text(data = dff %>% filter(connector=="to"),
            aes(label=node),stat="sankeynode",
            position=position_sankey(
              v_space="auto",order="ascending",nudge_x=-0.05),
            hjust=1,size=5,color="black") +
  scale_y_continuous(expand = expansion(mult=c(0.01,0))) +
  scale_x_discrete(expand = c(0,0)) +
  labs(x=NULL,y=NULL) +
  coord_cartesian(clip ="off") +
  theme(legend.position ="none",
        axis.text=element_blank(),
        axis.ticks = element_blank(),
        plot.background = element_blank(),
        panel.background = element_blank(),
        plot.margin = margin(0,0,0,5,unit="cm"))

df2 <- df %>%
  mutate(ymax = cumsum(Count)) %>% 
  mutate(ymin = ymax-Count) %>%
  mutate(label = (ymin + ymax)/2) 

p2 = ggplot(df2,aes(Log,label)) + 
  geom_point(aes(size=Count,color=enrichment_factor)) +
  scale_color_gradient(low="#00BFC4",high = "#F8766D") + 
  # geom_vline(xintercept=0)+
  labs(color="Enrichment",size="Count", shape="Category",
       x="-log10(P-value)",y="") + 
  theme_bw()+
  theme(axis.text = element_text(color = "black", size=13),
        axis.title = element_text(size=14),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.position = "top")


#####################################################################
############################# Fig. 5g ###############################
#####################################################################
col <- c("#F8766D","#7CAE00","#00BFC4","#C77CFF","#b395bd","yellow")
data_cancer = subset(data, subset=BC_type=="TNBC")
obj <- data_cancer
rd <- obj@reductions$umap@cell.embeddings
cl <- kmeans(rd,centers = 6)$cluster
tb <- as.matrix(table(cl[obj$CellType1=="CD8_N"]))
st <- rownames(tb)[tb==max(tb)]
tb <- as.matrix(table(cl[obj$CellType1=="CD8_EX2"]))
end1 <- rownames(tb)[tb==max(tb)]
tb <- as.matrix(table(cl[obj$CellType1=="CD8_RM"]))
end2 <- rownames(tb)[tb==max(tb)]
tb <- as.matrix(table(cl[obj$CellType1=="CD8_EMRA"]))
end3 <- rownames(tb)[tb==max(tb)]

lin1 <- getLineages(rd, cl, start.clus=as.numeric(st), end.clus=c(as.numeric(end1),as.numeric(end2),as.numeric(end3)))
crv1 <- getCurves(lin1)
plot(rd, col = col[as.factor(obj$CellType1)], pch=16, asp = 1, cex=0.5)
lines(SlingshotDataSet(crv1), lwd=2, col='black',cex=0.8)

DefaultAssay(data_cancer) = "RNA"
exp_time <- function(dir, gene){
  obj <- data_cancer
  df_batch <- c()
  for(i in 1:1){
    df_gene <- data.frame(pseudotime = crv1@assays@data@listData[["pseudotime"]][,i],
                          lineage = i,
                          expression = t(as.matrix(LayerData(obj,layer="data")))[,gene],
                          group = obj$sampletype) 
    df <- na.omit(df_gene)
    for(g in c("On","Pre")){
      df_tmp = df[df$group==g,]
      s <- smooth.spline(df_tmp$pseudotime,df_tmp$expression,df=3)
      cur <- data.frame(pseudotime=df_tmp$pseudotime,expression=predict(s,df_tmp$pseudotime)$y,lineage = i, group=g)
      df_batch <- rbind(df_batch,cur)
    }
  }
  df_batch$lineage_group <- paste(df_batch$lineage, df_batch$group, sep = "_")
  df_batch$lineage_group = factor(df_batch$lineage_group)
  p = ggplot(data=df_batch,aes(x=pseudotime,y=expression,group=lineage_group, color=lineage_group))+
    geom_line(size=1)+
    scale_y_continuous(expand = c(0,0))+
    scale_x_continuous(expand = c(0,0),breaks = c(0,5,10))+
    theme_classic()+
    theme(legend.position="none",axis.line = element_line(linewidth = 1.2),
          axis.text = element_text(size=25,color="black"),axis.title = element_text(size=0))
}
for(gene in c("HAVCR2","CXCL10","CXCL9")){
  exp_time(dir[1],gene)
}



