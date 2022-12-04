library(colorblindcheck)
library(pals)

# bivmes = function(pal){
#   tit = pal
#   pal = get(pal)
#   severity = 1
#   if (is.function(pal)){
#     pal = pal()
#     deu = colorspace::deutan(pal, severity = severity)
#     pro = colorspace::protan(pal, severity = severity)
#     tri = colorspace::tritan(pal, severity = severity)
#   }
#   return(list(tit,pal,deu,pro,tri))
#   
# }
bivmes2(arc.bluepink)
bivmes2 = function(pal){
  tit = pal
  severity = 1
  if (is.function(pal)){
    pal = pal()
    deu = colorspace::deutan(pal, severity = severity)
    pro = colorspace::protan(pal, severity = severity)
    tri = colorspace::tritan(pal, severity = severity)
  }
  return(list(tit,pal,deu,pro,tri))
  
}


opar<-par(no.readonly = T)

par(mfrow = c(3, 4), mar = c(1, 1, 2, 1))
bivcol(arc.bluepink)
bivcol(brewer.divdiv)
bivcol(brewer.divseq)
bivcol(brewer.qualseq)
bivcol(brewer.seqseq1)
bivcol(brewer.seqseq2)
bivcol(census.blueyellow)
bivcol(stevens.bluered)
bivcol(stevens.greenblue)
bivcol(stevens.pinkblue)
bivcol(stevens.pinkgreen)
bivcol(stevens.purplegold)
par(opar)
brewlist <- c('brewer.divdiv','brewer.divseq','brewer.qualseq',
              'brewer.seqseq1','brewer.seqseq2','census.blueyellow',
              'stevens.bluered','stevens.greenblue','stevens.pinkblue',
              'stevens.pinkgreen','stevens.purplegold')

df <- data.frame(matrix(NA, nrow = 9, ncol = 44))
for(i in 1:11){
  df[,(i-1)*4+1] <- bivmes(brewlist[i])[2]
  df[,(i-1)*4+2] <- bivmes(brewlist[i])[3]
  df[,(i-1)*4+3] <- bivmes(brewlist[i])[4]
  df[,(i-1)*4+4] <- bivmes(brewlist[i])[5]
  colnames(df)[(i-1)*4+1] <- bivmes(brewlist[i])[1]
  colnames(df)[(i-1)*4+2] <- paste(bivmes(brewlist[i])[1],'Deuteranopia',sep='-')
  colnames(df)[(i-1)*4+3] <- paste(bivmes(brewlist[i])[1],'Protanopia',sep='-')
  colnames(df)[(i-1)*4+4] <- paste(bivmes(brewlist[i])[1],'Tritanopia',sep='-')
}


write_csv(df, 'color_and_hexcode.csv')

df_o <- data.frame(matrix(nrow = 16,ncol =4))
colnames(df_o)[1] <- bivmes('arc.bluepink')[1]
colnames(df_o)[2] <- paste(bivmes('arc.bluepink')[1],'Deuteranopia',sep='-')
colnames(df_o)[3] <- paste(bivmes('arc.bluepink')[1],'Protanopia',sep='-')
colnames(df_o)[4] <- paste(bivmes('arc.bluepink')[1],'Tritanopia',sep='-')
df_o[,1] <-bivmes('arc.bluepink')[2]
df_o[,2] <-bivmes('arc.bluepink')[3]
df_o[,3] <-bivmes('arc.bluepink')[4]
df_o[,4] <-bivmes('arc.bluepink')[5]

write_csv(df_o, 'arc.bluepink.csv')


