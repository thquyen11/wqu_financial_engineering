library(shiny)
library(ggplot2)
library(readxl)
library(shinythemes)
require(gdata)


server <- function(input, output) {
  ###### NOTES & SCRIPT 2 - DATA CLEANING AND ANALYSIS #######
  ############################################################

  df <- read.xls("www/CountryData.xlsx")
  # Removing blank rows generated
  df<- df[c(1:21)]
  
  # Renaming the vairbles to be human readable :)
  names(df)[3:21]<- c("ForeignInvestment", "ElectricityAccess", "RenewableEnergy", "CO2Emission", "Inflation", "MobileSubscriptions", "InternetUse", "Exports", "Imports", "GDP", "MortalityMale", "MortalityFemale", "BirthRate", "DeathRate", "MortalityInfant", "LifeExpectancy", "FertilityRate", "PopulationGrowth", "UrbanPopulation")
  
  # checking correlations
  require(corrplot)
  output$correlation_plot <- renderPlot({
    cor_matrix = cor(df[,c(-1, -2, -grep(input$ignore_col, colnames(df)))],use="complete.obs")
    # cor_matrix = cor(df[,c(-1,-2, -grep(input$ignore_col, colnames(df)))],use="pairwise.complete.obs")
    corrplot(cor_matrix,method ="color",type="upper",tl.cex=0.7)
  })
  
  # Boxplot of standardised data
  output$boxplot_standardised_data <- renderPlot({
    boxplot(scale(df[,c(-1,-2, -grep(input$exclude_col, colnames(df)))]),las=2)
  })
  
  # Looking at null values
  # we check for NA's in rows
  null_rows = apply(df, 1, function(x) sum(is.na(x)))

  #we add the country names
  row_nulls = data.frame(df$CountryName,null_rows)

  #we select where not 0
  row_nulls[as.numeric(row_nulls[,2])>0,]

  #we check for nulls in columns
  apply(df, 2, function(x) sum(is.na(x)))

  # Setting seed so out results are imputation results are reprodcuible
  set.seed(0)

  require(caret)
  #we impute missing values with a random forest
  imputation_model = preProcess(x = df[,-c(1,2)],method = "bagImpute")
  imputated_data = predict(object = imputation_model,newdata=df[,-c(1,2)])

  # Adding country names to the rows
  rownames(imputated_data)<-df[,2]

  #we check for nulls in imputed data, success there are none :D
  apply(imputated_data, 2, function(x) sum(is.na(x)))


  ########### NOTES & SCRIPT 3 - MODEL BUILDING ##############
  ############################################################

  # PCA for our Biplot. Note scale =TRUE as we need to standardize our data
  pca.out<-prcomp(imputated_data,scale=TRUE)
  # pca.out
  # output$bipot_PCA <- renderPlot({
  #   biplot(pca.out,scale = 0, cex=0.75)
  # })


  # Creating a datatable to store and plot the
  # No of Principal Components vs Cumulative Variance Explained
  vexplained <- as.data.frame(pca.out$sdev^2/sum(pca.out$sdev^2))
  vexplained <- cbind(c(1:19),vexplained,cumsum(vexplained[,1]))
  colnames(vexplained) <- c("No_of_Principal_Components","Individual_Variance_Explained",
                            "Cumulative_Variance_Explained")

  
  # contributions
  library(factoextra)
  library(gridExtra)
  output$grid.arrange_contribution <- renderPlot({
    plot1 <- fviz_contrib(pca.out, choice="var", axes = 1, top = input$top_vari)
    plot2 <- fviz_contrib(pca.out, choice="var", axes = 2, top = input$top_vari, color = "lightgrey")
    grid.arrange(plot1, plot2, nrow=2)
  })
  

  # Scaling our data
  scaled_data = scale(imputated_data)

  # Visualising the dissimalirty matrix. Viually shows distinct clusters.
  output$fviz_dist_all <- renderPlot({
    fviz_dist(dist(scaled_data, method = input$dist_method), show_labels = FALSE)+ labs(title = "Distance Method")
  })

  
  # Choosing the number of k
  library(NbClust)
  res<- NbClust(scaled_data, distance = "euclidean", min.nc=2, max.nc=10,
                method = "kmeans", index = "all")


  ############## NOTES & SCRIPT 4 - Data Viz #################
  ############################################################
  
  # K-means clustering
  km.res <- kmeans(scaled_data, 3, nstart = 50)
  
  # Cluster visualisation
  output$cluster_visul <- renderPlot({
    fviz_cluster(km_reactive,
                 data = scaled_data,
                 palette = c("#2E9FDF", "#00AFBB", "#E7B800"),
                 ellipse.type = "euclid",
                 star.plot = TRUE, # Add segments from centroids to items
                 repel = TRUE, # Avoid label overplotting (slow)
                 ggtheme = theme_minimal()
    )
  })

  # Finding the averages for each cluster
  aggregate(imputated_data, by=list(cluster=km.res$cluster), mean)


  #Required libraries for world map plotting
  library(rworldmap)
  library(rworldxtra)

  cluster = as.numeric(km.res$cluster)

  #we plot to map
  par(mfrow=c(1,1))
  spdf = joinCountryData2Map(data.frame(cluster,df$CountryName), joinCode="NAME", nameJoinColumn="df.CountryName",verbose = TRUE,mapResolution = "low")
  output$mapCountryData <- renderPlot({
    mapCountryData(spdf, nameColumnToPlot="cluster", catMethod="fixedWidth",colourPalette=c(input$color_clust1,input$color_clust2,input$color_clust3), addLegend = FALSE, lwd = 0.5)
  })
  
}


ui <- fluidPage(theme = "bootstrap.min_slate.css",
                br(),
                br(),
  column(3, offset=4, titlePanel("Week 1 Assignment")),
  br(),
  column(3, offset=4, titlePanel("Clustering Analysis on Countries")),
  br(),
  br(),
  br(),
  br(),
  br(),
  
  fluidRow(
    column(4,
           sidebarPanel(
             selectInput('ignore_col', 'Ignore Variable', names(df)),
           ),       
    ),
     column(8,
            mainPanel(
              plotOutput('correlation_plot'),
            )
    )
  ),
  
  fluidRow(
    column(4,
           sidebarPanel(
             selectInput('exclude_col', 'Excluding the effect of...', names(df)),
           ),       
    ),
    column(8,
           mainPanel(
             plotOutput('boxplot_standardised_data'),
           )
    )
  ),
  
  fluidRow(
    column(4,
           sidebarPanel(
             sliderInput('top_vari', 'Number of Variables', min=2, max=20,
                         value=min(14, 20), step=2, round=0),
           ),       
    ),
    column(8,
           mainPanel(
             plotOutput('grid.arrange_contribution'),
           )
    )
  ),
  
  fluidRow(
    column(4,
           sidebarPanel(
             selectInput('dist_method', 'Distance Method', c('euclidean','maximum','manhattan','canberra','binary','minkowski'))
           ),       
    ),
    column(8,
           mainPanel(
             plotOutput('fviz_dist_all'),
           )
    )
  ),
  
  
  fluidRow(
    column(4,
           sidebarPanel(
             selectInput('color_clust1', 'Cluster1 Color', c("#FFFF00","#000000","#FFFFFF","#FF0000","#00FF00","#0000FF","#00FFFF","#FF00FF","#C0C0C0","#808080","#800000","#808000","#008000","#800080","#008080","#000080")),
             selectInput('color_clust2', 'Cluster2 Color', c("#00FF00","#000000","#FFFFFF","#FF0000","#0000FF","#FFFF00","#00FFFF","#FF00FF","#C0C0C0","#808080","#800000","#808000","#008000","#800080","#008080","#000080")),
             selectInput('color_clust3', 'Cluster3 Color', c("#800000","#000000","#FFFFFF","#FF0000","#00FF00","#0000FF","#FFFF00","#00FFFF","#FF00FF","#C0C0C0","#808080","#808000","#008000","#800080","#008080","#000080")),
             
             ),       
    ),
    column(8,
           mainPanel(
             plotOutput('mapCountryData'),
           )
    )
  ),
)


shinyApp(ui = ui, server = server)