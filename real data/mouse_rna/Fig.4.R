pkgs <- c("ggplot2","patchwork","readxl","UpSetR","data.table","Seurat","dplyr",
          "Rtsne","uwot","cluster","tibble","ggrepel","org.Mm.eg.db","tidyr",
          "clusterProfiler","ReactomePA","stringr","ggsankeyfier","RColorBrewer")
for (pkg in pkgs){
  suppressPackageStartupMessages(library(pkg, character.only = T))
}
#####################################################################
############################# Fig. 4b ###############################
#####################################################################
p = ggplot(metric, aes(x=Method,color=Method,fill=Method))+
  geom_bar(aes(y=Accuracy), stat="identity", width = 0.6)+
  scale_color_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  scale_fill_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  scale_y_continuous(limits = c(0,1.15),breaks = c(0,0.5,1))+
  theme_bw()+
  theme(axis.text = element_text(size=13, color = "black"),
        axis.title = element_text(size=14),
        panel.grid = element_blank())


#####################################################################
############################# Fig. 4c ###############################
#####################################################################
load("selected_features.Rdata")
features = selected_features
res <- list()
for(i in 1:5){
  feature = na.omit(features[,i])
  res[[colnames(features)[i]]] <- feature
}
upset(fromList(res),
      text.scale=c(2,2,2,2,2,2))


#####################################################################
############################# Fig. 4d ###############################
#####################################################################
X = read.table("chen_X.txt")
X = as.data.frame(X)
gene = read.table("gene_names.csv")
rownames(X) = gene$V1
colnames(X) = paste0("cell",c(1:5282))
Y = read.table("chen_Y.txt")
Y = as.data.frame(Y)
Y$celltype = "OPC"
Y$celltype[Y$V1==1] = "MO"

data1 <- X[,which(Y$celltype=="OPC")]
data2 <- X[,which(Y$celltype=="MO")]
data = cbind(data1,data2)

sc_metric <- matrix(NA, nrow = 1,ncol = 5)
colnames(sc_metric) = colnames(features)
p = list()
for(i in 1:5){
  data_tmp = data[rownames(data)%in%c(features[,i]),]
  seu <- CreateSeuratObject(counts = data_tmp)
  seu$group = c(rep("OPC", 1741), rep("MO", 3541))
  
  if(nrow(data_tmp)>30){
    seu <- seu %>%
      NormalizeData(normalization.method = "LogNormalize",scale.factor = 10000) %>%
      FindVariableFeatures(selection.method = "vst", nfeatures = 2000) %>%
      ScaleData() %>%
      RunPCA(verbose = FALSE) %>%
      FindNeighbors(reduction = "pca", dims = 1:30) %>%
      RunUMAP(reduction = "pca",dims = 1:30) %>%
      RunTSNE(reduction = "pca", dims = 1:30)
    p[[i]] = DimPlot(seu,reduction = "tsne",group.by = "group", cols = c("#ffd4b7","#d0b4d5"))
    tsne_df <- data.frame(seu@reductions$tsne@cell.embeddings)
  }else{
    seu <- seu %>%
      NormalizeData(normalization.method = "LogNormalize",scale.factor = 10000) %>%
      ScaleData()
    dat = as.matrix(LayerData(seu, layer = "data"))
    tsn <- Rtsne(t(dat), check_duplicates=FALSE)
    df <- as.data.frame(tsn$Y)
    tsne_df = df
    df$group <- c(rep("OPC", 1741), rep("MO", 3541))
    p[[i]] = ggplot(data = df,aes(x = V1, y = V2, color=group))+
      geom_point(size = 1)+
      scale_color_manual(values=c("#ffd4b7","#d0b4d5"))+
      theme_classic()+
      xlab("tSNE_1")+
      ylab("tSNE_2")+
      theme(axis.text = element_text(size=13, color = "black"),
            axis.title = element_text(size=14),
            panel.grid = element_blank())
  }
  dis <- dist(tsne_df)^2
  type <- as.numeric(factor(c(rep("OPC", 1741), rep("MO", 3541))))
  sc_metric[i] <- mean(silhouette(type, dis)[,3])
}
sc_metric = round(sc_metric,2)


#####################################################################
############################# Fig. 4e ###############################
#####################################################################
seu <- CreateSeuratObject(counts = data)
seu$group = c(rep("OPC", 1741), rep("MO", 3541))
seu <- seu %>%
  NormalizeData(normalization.method = "LogNormalize",scale.factor = 10000) %>%
  FindVariableFeatures(selection.method = "vst", nfeatures = 2000) %>%
  ScaleData() %>%
  RunPCA(verbose = FALSE) %>%
  FindNeighbors(reduction = "pca", dims = 1:20) %>%
  RunUMAP(reduction = "pca",dims = 1:20) %>%
  RunTSNE(reduction = "pca", dims = 1:20)
Idents(seu) <- "group"
markers <- FindMarkers(seu, ident.1 = "OPC",ident.2 = "MO")
logFC_t <- mean(markers$avg_log2FC)+2*sd(markers$avg_log2FC)
nmarker <- markers %>%
  mutate(change = as.factor(ifelse(p_val > 0.05,"stable",
                                   ifelse(avg_log2FC>logFC_t,"up",
                                          ifelse(avg_log2FC < -logFC_t ,'down','stable')))))%>%
  rownames_to_column('gene')
marker <- nmarker$gene[nmarker$change%in%c("up","down")]

marker_num <- matrix(NA, nrow = 2,ncol = 5)
colnames(marker_num) = colnames(features)
for(i in 1:5){
  marker_num[1,i] = sum(features[,i]%in%marker)
  marker_num[2,i] = length(na.omit(features[,i]))
}
df_deg = data.frame(method=colnames(marker_num),num=marker_num[1,],type="DEG")
df_nondeg = data.frame(method=colnames(marker_num),num=marker_num[2,]-marker_num[1,],type="non-DEG")
marker_num <- rbind(df_deg,df_nondeg)
marker_num$method <- factor(marker_num$method, levels = c("deepvaris","survnet","DeepLINK","HiDe","stabl"))

marker_num$type = factor(marker_num$type, levels = c("non-DEG","DEG"))
p = ggplot(marker_num, aes(x=method,y=num,fill=type,color=method))+
  geom_bar(stat="identity", position="stack", width = 0.6, linewidth = 1)+
  scale_color_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  # scale_fill_manual(values = c("#D55276","#699eca","#b395bd","#98D4AB","#00BFC4"))+
  scale_fill_manual(values = c("non-DEG" = "white","DEG" = "steelblue")) +
  theme_bw()+
  theme(axis.text = element_text(size=13, color = "black"),
        axis.title = element_text(size=14),
        panel.grid = element_blank())


#####################################################################
############################# Fig. 4f ###############################
#####################################################################
gene = setdiff(features$deepvaris,marker)

g_co = c()
marker = data.frame(gene=gene)
ids = bitr(marker$gene,'SYMBOL','ENTREZID',"org.Mm.eg.db") 
cutoff <- 0.05

m1 <- merge(marker,ids,by.x='gene',by.y='SYMBOL')
go1up <- enrichGO(gene = m1$ENTREZID, 
                  OrgDb = org.Mm.eg.db,  
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
  df1up$method = "DeepVaris"
}
g_co <- rbind(g_co, df1up)

df <- g_co[grep("oligo", g_co$Description), ]
df$enrichment_factor <- 0
for(i in 1:nrow(df)){    
  df[i,]$enrichment_factor <- eval(parse(text = df[i,]$GeneRatio))/eval(parse(text = df[i,]$BgRatio))
}
df <- df[order(df$Log),]
df <- df %>% distinct(Description, .keep_all = TRUE)
df$Description <- factor(df$Description, levels = df$Description)
df_sep = df %>%
  separate_rows(.,geneID,convert=TRUE,sep="/")
ids = bitr(df_sep$geneID,'ENTREZID','SYMBOL',"org.Mm.eg.db") 
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
  labs(color="Enrichment",size="Count", shape="Category",
       x="-log10(P-value)",y="") + 
  theme_bw()+
  theme(axis.text = element_text(color = "black", size=13),
        axis.title = element_text(size=14),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.position = "top")


