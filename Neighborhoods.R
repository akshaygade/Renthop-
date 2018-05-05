library(tigris)
library(dplyr)
library(leaflet)
library(sp)
library(ggmap)
library(maptools)
library(broom)
library(httr)
library(rgdal)

r <- GET('http://data.beta.nyc//dataset/0ff93d2d-90ba-457c-9f7e-39e47bf2ac5f/resource/35dd04fb-81b3-479b-a074-a27a37888ce7/download/d085e2f8d0b54d4590b1e7d1f35594c1pediacitiesnycneighborhoods.geojson')
nyc_neighborhoods <- readOGR(content(r,'text'), 'OGRGeoJSON', verbose = F)
nyc_neighborhoods_df <- tidy(nyc_neighborhoods)

nyc_neighborhoods_df <- tidy(nyc_neighborhoods)
nyc_map <- get_map(location = c(lon = -74.00, lat = 40.71), maptype = "terrain", zoom = 11)
suppressMessages(ggmap(nyc_map)) + 
  geom_polygon(data=nyc_neighborhoods_df, aes(x=long, y=lat, group=group), color="blue", fill=NA)

#Finding Neighborhoods of all locations
lats <- train$latitude
lngs <- train$longitude
points <- data.frame(lat=as.numeric(lats), lng=as.numeric(lngs))
points_spdf <- points
coordinates(points_spdf) <- ~lng + lat
proj4string(points_spdf) <- proj4string(nyc_neighborhoods)
matches <- over(points_spdf, nyc_neighborhoods)
points <- cbind(points, matches)

#Plotting the distribution of listings
points <- train[c('lat','lng','neighborhood','boroughCode','borough','X.id')]
points_by_neighborhood <- points %>%
  group_by(neighborhood) %>%
  summarize(num_points=n())

map_data <- geo_join(nyc_neighborhoods, points_by_neighborhood, "neighborhood", "neighborhood")
pal <- colorNumeric(palette = "RdBu", domain = range(map_data@data$num_points, na.rm=T))

plot_data <- tidy(nyc_neighborhoods, region="neighborhood") %>%
  left_join(., points_by_neighborhood, by=c("id"="neighborhood")) %>%
  filter(!is.na(num_points))
nyc_map <- get_map(location = c(lon = -74.00, lat = 40.71), maptype = "terrain", zoom = 10)

ggmap(nyc_map) + 
  geom_polygon(data=plot_data, aes(x=long, y=lat, group=group, fill=num_points),colour='black', alpha=0.75)

#Exploring transportation of each listing
