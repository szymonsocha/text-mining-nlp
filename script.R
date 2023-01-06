################################################################################
# Text Mining and Social Media Mining - Project
#
# Clustering and Topic Modelling 
#
################################################################################

# Libraries
library(tm)
library(wordcloud)
library(factoextra)
library(NbClust)
library(cluster)
library(dplyr)
library(lubridate)
library(stringr)
library(textmineR)

set.seed(2023)


# Data source:
# https://www.kaggle.com/datasets/therohk/million-headlines

raw_data <- read.csv('data/abcnews-date-text.csv') %>%
  mutate(publish_date = ymd(publish_date))

raw_data <- raw_data %>% 
  tibble::rownames_to_column(var = "id")

data_2004 <- raw_data %>%
  filter(between(publish_date, as.Date("2004-01-01"), as.Date("2004-12-31"))) %>% 
  # since the source data is from Australia, remove this word from the dataset
  mutate(across('headline_text', str_replace, 'australia', '')) %>% 
  mutate(across('headline_text', str_replace, 'australian', ''))

data_2020 <- raw_data %>%
  filter(between(publish_date, as.Date("2020-01-01"), as.Date("2020-12-31"))) %>% 
  # consider covid as the same thing as coronavirus
  mutate(across('headline_text', str_replace, 'covid', 'coronavirus'))


# Text Clustering --------------------------------------------------------------

## 2004 ----

docs_2004 <- VCorpus(VectorSource(data_2004$headline_text))

# Preliminary cleaning, Cleaning text and Stopword removal ----
toSpace <- content_transformer(function (x, pattern) gsub(pattern, " ", x))

docs_2004 <- tm_map(docs_2004, removePunctuation)
docs_2004 <- tm_map(docs_2004, removeNumbers)
docs_2004 <- tm_map(docs_2004, content_transformer(tolower))
docs_2004 <- tm_map(docs_2004, removeWords, stopwords("english"))
docs_2004 <- tm_map(docs_2004, removeWords, c("australia", "australian", "nsw", "queensland", "victoria")) # since the source data is from Australia, remove this word from the dataset
docs_2004 <- tm_map(docs_2004, stripWhitespace)


# Create term-document matrix
tdm_2004 <- TermDocumentMatrix(docs_2004)
tdm_2004


# Reduce sparcity
tdms_2004 <- removeSparseTerms(tdm_2004, sparse = 0.99)
tdms_2004

# Calculate distance matrix
best_n_matrix_2004 <- as.matrix(dist(tdms_2004, method="euclidian"))

## Elbow method
fviz_nbclust(best_n_matrix_2004, kmeans, method = "wss") +
  geom_vline(xintercept = 2, linetype = "dashed") +
  labs(subtitle = "Elbow method")

## Silhouette method
fviz_nbclust(best_n_matrix_2004, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")



kfit_2004 <- kmeans(dist(tdms_2004, method="euclidian"), 4)

clusplot(as.matrix(dist(tdms_2004, method="euclidian")), kfit_2004$cluster, color=T, shade=T, labels=2, lines=0, 
         main = "2D Representation of Clusters")




d_2004 <- dist(tdms_2004, method="euclidian")

hc_2004 <- hclust(d_2004, "ward.D")


# Hierarchical
#clustering <- cutree(hc, 4)

# Kmeans
clustering_2004 <- kfit_2004$cluster


plot(hc_2004, main = "Hierarchical clustering",
     ylab = "", xlab = "", yaxt = "n")




p_words_2004 <- colSums(t(as.matrix(tdms_2004))) / sum(t(as.matrix(tdms_2004)))

cluster_words_2004 <- lapply(unique(clustering_2004), function(x){
  rows_2004 <- tdms_2004[ clustering_2004 == x , ]
  
  # for memory's sake, drop all words that don't appear in the cluster
  rows_2004 <- rows_2004[ , colSums(t(as.matrix(rows_2004))) > 0 ]
  
  colSums(t(as.matrix(rows_2004))) / sum(t(as.matrix(rows_2004))) - p_words_2004[ colnames(t(as.matrix(rows_2004))) ]
})



# create a summary table of the top 5 words defining each cluster
cluster_summary_2004 <- data.frame(cluster = unique(clustering_2004),
                                   size = as.numeric(table(clustering_2004)),
                                   top_words = sapply(cluster_words_2004, function(d){
                                     paste(
                                       names(d)[ order(d, decreasing = TRUE) ][ 1:5 ], 
                                       collapse = ", ")
                                   }),
                                   stringsAsFactors = FALSE)

cluster_summary_2004



wordcloud(words = names(cluster_words_2004[[ 1 ]]), 
          freq = cluster_words_2004[[ 1 ]], max.words=5, rot.per=0.2, colors = brewer.pal(6, "Dark2"))

wordcloud(words = names(cluster_words_2004[[ 2 ]]), 
          freq = cluster_words_2004[[ 2 ]], max.words=5, rot.per=0.2, colors = brewer.pal(6, "Dark2"))



## 2020 ----

docs_2020 <- VCorpus(VectorSource(data_2020$headline_text))

# Preliminary cleaning, Cleaning text and Stopword removal ----
toSpace <- content_transformer(function (x, pattern) gsub(pattern, " ", x))

docs_2020 <- tm_map(docs_2020, removePunctuation)
docs_2020 <- tm_map(docs_2020, removeNumbers)
docs_2020 <- tm_map(docs_2020, content_transformer(tolower))
docs_2020 <- tm_map(docs_2020, removeWords, stopwords("english"))
docs_2020 <- tm_map(docs_2020, removeWords, c("australia", "australian", "nsw", "queensland", "victoria")) # since the source data is from Australia, remove this word from the dataset
docs_2020 <- tm_map(docs_2020, stripWhitespace)


# Create term-document matrix
tdm_2020 <- TermDocumentMatrix(docs_2020)
tdm_2020


# Reduce sparcity
tdms_2020 <- removeSparseTerms(tdm_2020, sparse = 0.99)
tdms_2020

# Calculate distance matrix
best_n_matrix_2020 <- as.matrix(dist(tdms_2020, method="euclidian"))

## Elbow method
fviz_nbclust(best_n_matrix_2020, kmeans, method = "wss") +
  geom_vline(xintercept = 2, linetype = "dashed") +
  labs(subtitle = "Elbow method")

## Silhouette method
fviz_nbclust(best_n_matrix_2020, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")



kfit_2020 <- kmeans(dist(tdms_2020, method="euclidian"), 2)

clusplot(as.matrix(dist(tdms_2020, method="euclidian")), kfit_2020$cluster, color=T, shade=T, labels=2, lines=0, 
         main = "2D Representation of Clusters")




d_task2_2020 <- dist(tdms_2020, method="euclidian")

hc_2020 <- hclust(d_task2_2020, "ward.D")


# Hierarchical
#clustering <- cutree(hc, 4)

# Kmeans
clustering_2020 <- kfit_2020$cluster


plot(hc_2020, main = "Hierarchical clustering",
     ylab = "", xlab = "", yaxt = "n")




p_words_task2 <- colSums(t(as.matrix(tdms_2020))) / sum(t(as.matrix(tdms_2020)))

cluster_words_2020 <- lapply(unique(clustering_2020), function(x){
  rows_2020 <- tdms_2020[ clustering_2020 == x , ]
  
  # for memory's sake, drop all words that don't appear in the cluster
  rows_2020 <- rows_2020[ , colSums(t(as.matrix(rows_2020))) > 0 ]
  
  colSums(t(as.matrix(rows_2020))) / sum(t(as.matrix(rows_2020))) - p_words_task2[ colnames(t(as.matrix(rows_2020))) ]
})



# create a summary table of the top 5 words defining each cluster
cluster_summary_2020 <- data.frame(cluster = unique(clustering_2020),
                              size = as.numeric(table(clustering_2020)),
                              top_words = sapply(cluster_words_2020, function(d){
                                paste(
                                  names(d)[ order(d, decreasing = TRUE) ][ 1:5 ], 
                                  collapse = ", ")
                              }),
                              stringsAsFactors = FALSE)

cluster_summary_2020



wordcloud(words = names(cluster_words_2020[[ 1 ]]), 
          freq = cluster_words_2020[[ 1 ]], max.words=5, rot.per=0.2, colors = brewer.pal(6, "Dark2"))

wordcloud(words = names(cluster_words_2020[[ 2 ]]), 
          freq = cluster_words_2020[[ 2 ]], max.words=5, rot.per=0.2, colors = brewer.pal(6, "Dark2"))


### Remove 'coronavirus' word ----

# Coronavirus is defininitely dominating in 2020, let's remove it from the analysis and compare the results

#docs_2020 <- VCorpus(VectorSource(data_2020$headline_text))

# Preliminary cleaning, Cleaning text and Stopword removal ----
#toSpace <- content_transformer(function (x, pattern) gsub(pattern, " ", x))

#docs_2020 <- tm_map(docs_2020, removePunctuation)
#docs_2020 <- tm_map(docs_2020, removeNumbers)
#docs_2020 <- tm_map(docs_2020, content_transformer(tolower))
#docs_2020 <- tm_map(docs_2020, removeWords, stopwords("english"))
#docs_2020 <- tm_map(docs_2020, removeWords, c("australia", "australian")) # since the source data is from Australia, remove this word from the dataset
docs_2020 <- tm_map(docs_2020, removeWords, c("coronavirus")) # remove 'coronavirus'
#docs_2020 <- tm_map(docs_2020, stripWhitespace)


# Create term-document matrix
tdm_2020 <- TermDocumentMatrix(docs_2020)
tdm_2020


# Reduce sparcity
tdms_2020 <- removeSparseTerms(tdm_2020, sparse = 0.99)
tdms_2020

# Calculate distance matrix
best_n_matrix_2020 <- as.matrix(dist(tdms_2020, method="euclidian"))

## Elbow method
fviz_nbclust(best_n_matrix_2020, kmeans, method = "wss") +
  geom_vline(xintercept = 2, linetype = "dashed") +
  labs(subtitle = "Elbow method")

## Silhouette method
fviz_nbclust(best_n_matrix_2020, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")



kfit_2020 <- kmeans(dist(tdms_2020, method="euclidian"), 2)

clusplot(as.matrix(dist(tdms_2020, method="euclidian")), kfit_2020$cluster, color=T, shade=T, labels=2, lines=0, 
         main = "2D Representation of Clusters")




d_task2_2020 <- dist(tdms_2020, method="euclidian")

hc_2020 <- hclust(d_task2_2020, "ward.D")


# Hierarchical
#clustering <- cutree(hc, 4)

# Kmeans
clustering_2020 <- kfit_2020$cluster


plot(hc_2020, main = "Hierarchical clustering",
     ylab = "", xlab = "", yaxt = "n")




p_words_task2_2020 <- colSums(t(as.matrix(tdms_2020))) / sum(t(as.matrix(tdms_2020)))

cluster_words_2020 <- lapply(unique(clustering_2020), function(x){
  rows_2020 <- tdms_2020[ clustering_2020 == x , ]
  
  # for memory's sake, drop all words that don't appear in the cluster
  rows_2020 <- rows_2020[ , colSums(t(as.matrix(rows_2020))) > 0 ]
  
  colSums(t(as.matrix(rows_2020))) / sum(t(as.matrix(rows_2020))) - p_words_task2[ colnames(t(as.matrix(rows_2020))) ]
})



# create a summary table of the top 5 words defining each cluster
cluster_summary_2020 <- data.frame(cluster = unique(clustering_2020),
                                   size = as.numeric(table(clustering_2020)),
                                   top_words = sapply(cluster_words_2020, function(d){
                                     paste(
                                       names(d)[ order(d, decreasing = TRUE) ][ 1:5 ], 
                                       collapse = ", ")
                                   }),
                                   stringsAsFactors = FALSE)

cluster_summary_2020



wordcloud(words = names(cluster_words_2020[[ 1 ]]), 
          freq = cluster_words_2020[[ 1 ]], max.words=5, rot.per=0.2, colors = brewer.pal(6, "Dark2"))

wordcloud(words = names(cluster_words_2020[[ 2 ]]), 
          freq = cluster_words_2020[[ 2 ]], max.words=5, rot.per=0.2, colors = brewer.pal(6, "Dark2"))



# Topic Modelling --------------------------------------------------------------

library(tidyverse) # general utility & workflow functions
library(tidytext) # tidy implimentation of NLP methods
library(topicmodels) # for LDA topic modelling 
library(tm) # general text mining functions, making document term matrixes
library(SnowballC) # for stemming



top_terms_by_topic_LDA <- function(input_text, # should be a columm from a dataframe
                                   plot = T, # return a plot? TRUE by defult
                                   number_of_topics = 4) # number of topics (4 by default)
{    
  # create a corpus (type of object expected by tm) and document term matrix
  Corpus <- Corpus(VectorSource(input_text)) # make a corpus object
  DTM <- DocumentTermMatrix(Corpus) # get the count of words/document
  
  # remove any empty rows in our document term matrix (if there are any 
  # we'll get an error when we try to run our LDA)
  unique_indexes <- unique(DTM$i) # get the index of each unique value
  DTM <- DTM[unique_indexes,] # get a subset of only those indexes
  
  # preform LDA & get the words/topic in a tidy text format
  lda <- LDA(DTM, k = number_of_topics, control = list(seed = 1234))
  topics <- tidy(lda, matrix = "beta")
  
  # get the top ten terms for each topic
  top_terms <- topics  %>% # take the topics data frame and..
    group_by(topic) %>% # treat each topic as a different group
    top_n(10, beta) %>% # get the top 10 most informative words
    ungroup() %>% # ungroup
    arrange(topic, -beta) # arrange words in descending informativeness
  
  # if the user asks for a plot (TRUE by default)
  if(plot == T){
    # plot the top ten terms for each topic in order
    top_terms %>% # take the top terms
      mutate(term = reorder(term, beta)) %>% # sort terms by beta value 
      ggplot(aes(term, beta, fill = factor(topic))) + # plot beta by theme
      geom_col(show.legend = FALSE) + # as a bar plot
      facet_wrap(~ topic, scales = "free") + # which each topic in a seperate plot
      labs(x = NULL, y = "Beta") + # no x label, change y label 
      coord_flip() # turn bars sideways
  }else{ 
    # if the user does not request a plot
    # return a list of sorted terms instead
    return(top_terms)
  }
}


## 2004 ----

# create a document term matrix to clean
Corpus_2004 <- Corpus(VectorSource(data_2004$headline_text)) 
DTM_2004 <- DocumentTermMatrix(Corpus_2004)

# convert the document term matrix to a tidytext corpus
DTM_tidy_2004 <- tidy(DTM_2004)

# I'm going to add my own custom stop words that I don't think will be
# very informative
custom_stop_words_2004 <- tibble(word = c("australia", "australian",
                                          "queensland", "nsw", "victoria"))

# remove stopwords
DTM_tidy_cleaned_2004 <- DTM_tidy_2004 %>% # take our tidy dtm and...
  anti_join(stop_words, by = c("term" = "word")) %>% # remove English stopwords and...
  anti_join(custom_stop_words_2004, by = c("term" = "word")) # remove my custom stopwords

# reconstruct cleaned documents (so that each word shows up the correct number of times)
# stem the words (e.g. convert each word to its stem, where applicable)
cleaned_documents_2004 <- DTM_tidy_cleaned_2004 %>%
  group_by(document) %>% 
  mutate(terms = toString(rep(wordStem(term), count))) %>%
  select(document, terms) %>%
  unique()

# now let's look at the new most informative terms
top_terms_by_topic_LDA(cleaned_documents_2004$terms, number_of_topics = 8)


## TESTS
if(0){
  Corpus_2004 <- Corpus(VectorSource(data_2004$headline_text)) 
  DTM_2004 <- DocumentTermMatrix(Corpus_2004)
  
  # convert the document term matrix to a tidytext corpus
  DTM_tidy_2004 <- tidy(DTM_2004)
  
  # I'm going to add my own custom stop words that I don't think will be
  # very informative
  custom_stop_words_2004 <- tibble(word = c("australia", "australian",
                                            "queensland", "nsw", "victoria"))
  
  # remove stopwords
  DTM_tidy_cleaned_2004 <- DTM_tidy_2004 %>% # take our tidy dtm and...
    anti_join(stop_words, by = c("term" = "word")) %>% # remove English stopwords and...
    anti_join(custom_stop_words_2004, by = c("term" = "word")) # remove my custom stopwords
  
  # reconstruct cleaned documents (so that each word shows up the correct number of times)
  # stem the words (e.g. convert each word to its stem, where applicable)
  cleaned_documents_2004 <- DTM_tidy_cleaned_2004 %>%
    group_by(document) %>% 
    mutate(terms = toString(rep(wordStem(term), count))) %>%
    select(document, terms) %>%
    unique()
  
  # remove any empty rows in our document term matrix (if there are any 
  # we'll get an error when we try to run our LDA)
  unique_indexes <- unique(DTM_2004$i) # get the index of each unique value
  DTM_2004 <- DTM_2004[unique_indexes,] # get a subset of only those indexes
  
  # preform LDA & get the words/topic in a tidy text format
  lda <- LDA(DTM_2004, k = 50, control = list(seed = 1234))
  topics <- tidy(lda, matrix = "beta")
  
  
  library(lda)
  library(reshape2)
  
  tmResult <- posterior(lda)
  theta <- tmResult$topics
  beta <- tmResult$terms
  topicNames <- apply(terms(lda, 5), 2, paste, collapse = " ")  # reset topicnames
  
  exampleIds <- c(786, 4632, 9876)
  N <- length(exampleIds)
  
  topicProportionExamples <- theta[exampleIds,]
  colnames(topicProportionExamples) <- topicNames
  vizDataFrame <- melt(cbind(data.frame(topicProportionExamples), document = factor(1:N)), variable.name = "topic", id.vars = "document")  
  ggplot(data = vizDataFrame, aes(topic, value, fill = document), ylab = "proportion") + 
    geom_bar(stat="identity") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +  
    coord_flip() +
    facet_wrap(~ document, ncol = N)
  
  topicNames <- apply(lda::top.topic.words(beta, 5, by.score = T), 2, paste, collapse = " ")
  
  Corpus_2004$content[685]
  data_2004
  data_2004[data_2004$id == 7654,]
}
###





## 2020 ----

# create a document term matrix to clean
Corpus_2020 <- Corpus(VectorSource(data_2020$headline_text)) 
DTM_2020 <- DocumentTermMatrix(Corpus_2020)

# convert the document term matrix to a tidytext corpus
DTM_tidy_2020 <- tidy(DTM_2020)

# I'm going to add my own custom stop words that I don't think will be
# very informative
custom_stop_words_2020 <- tibble(word = c("australia", "australian", 
                                          "queensland", "nsw", "victoria",
                                          "coronavirus"))

# remove stopwords
DTM_tidy_cleaned_2020 <- DTM_tidy_2020 %>% # take our tidy dtm and...
  anti_join(stop_words, by = c("term" = "word")) %>% # remove English stopwords and...
  anti_join(custom_stop_words_2020, by = c("term" = "word")) # remove my custom stopwords

# reconstruct cleaned documents (so that each word shows up the correct number of times)
# stem the words (e.g. convert each word to its stem, where applicable)
cleaned_documents_2020 <- DTM_tidy_cleaned_2020 %>%
  group_by(document) %>% 
  mutate(terms = toString(rep(wordStem(term), count))) %>%
  select(document, terms) %>%
  unique()

# now let's look at the new most informative terms
top_terms_by_topic_LDA(cleaned_documents_2020$terms, number_of_topics = 2)
