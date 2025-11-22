library(ppclust)
library(factoextra)
library(dplyr)
library(cluster)
library(fclust)
library(psych)
library(clusterSim)
library(corrplot)
library(REdaS)

#Membaca data
head(PMA_Investasi2)

#Meremove kolom Negara
data_selected = PMA_Investasi2[,-1]
head(data_selected)
str(data_selected)

#cek missing value
any(is.na(data_selected))

#CEK DUPLIKAT
any(duplicated(data_selected))

#CEK OUTLIER
boxplot(scale(data_selected))

#WINSORIZING UNTUK MENGATASI OUTLIER
data_selected <- as.data.frame(data_selected)

for (i in seq_along(data_selected)) {
  if (is.numeric(data_selected[[i]]))
    x <- data_selected[[i]]
  
  qnt <- quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)
  caps <- quantile(x, probs = c(0.05, 0.95), na.rm = TRUE)
  H <- 1.5 * IQR(x, na.rm = TRUE)
  
  x[x < (qnt[1] - H)] <- caps[1]
  x[x > (qnt[2] + H)] <- caps[2]
  
  data_selected[[i]] <- x
}

#BOXPLOT SETELAH WINSORIZING
boxplot(scale(data_selected), main = "Boxplot Setelah Winsorizing")

#NORMALISASI MENGGUNAKAN Z SCORE
data_normalisasi <- as.data.frame(scale(data_selected))
boxplot(data_normalisasi)

#PLOT KORELASI (CEK MULTIKOLINEARITAS)
corrplot(cor(data_normalisasi),method="ellipse",type="upper")

#bartlett test of sphericity
bart_spher(data_normalisasi)

#KAISER MAYER OLKIN (MEREDUKSI ATRIBUT YANG TIDAK MEMENUHI <0.5)
KMO(data_normalisasi)

#PCA
pca <- princomp(data_normalisasi, cor=FALSE)
screeplot(pca,type="lines",col=4)

#MENYIMPAN HASIL PCA
data_selected_pca <- pca$scores[,1:2]

#ELBOW
fviz_nbclust(data_selected_pca,kmeans,method = "wss")

#SILHOUETTE
fviz_nbclust(data_selected_pca,kmeans,method = "silhouette")


#KMEANS
kmRes <- kmeans(data_selected_pca,2)
table(kmRes$cluster)

fviz_cluster(
  object = kmRes,
  data = data_selected_pca,
  geom = c("point","text","centroid"),
  ellipse.type = "convex",
  palette = "jco",
  repel = TRUE,
  ggtheme = theme_minimal()
)

#DAVIES BOULDIN INDEX
dbi_value <- index.DB(x = as.matrix(data_selected_pca), cl = kmRes$cluster, centrotypes = "centroids")$DB
cat("Davies-Bouldin Index (DBI)=", dbi_value)

#HASIL CENTROID
kmRes$centers

#MENAMBAHKAN HASIL CLUSTER KE DATA AWAL
hasil_cluster <- data.frame(PMA_Investasi2, Cluster = kmRes$cluster)
head(hasil_cluster)

#MENGGABUNGKAN DENGAN DATA_SELECTED_PCA
negara <- PMA_Investasi2$Negara

data_clustered_full <- cbind(Negara = negara, data_selected_pca, Cluster = kmRes$cluster)

head(data_clustered_full)

write.csv(data_clustered_full,"hasil_cluster.csv", row.names = FALSE)

getwd()