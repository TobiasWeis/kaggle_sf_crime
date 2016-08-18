library(ggmap)

minlon <- c(-122.52469)
maxlon <- c(-122.33663)
minlat <- c(37.69862)
maxlat <- c(37.82986)

mylocation <- c(minlon, minlat, maxlon, maxlat)
# sources   types
# stamen    watercolor, toner, terrain
# osm       
# google    terrain, satellite, roadmap, hybrid
mymap <- ggmap(get_map(location=mylocation, source="osm", zoom=13), extent="device")
# this line creates a plot with white borders?!
# have to use the temp-file created by ggmap
ggsave("/home/weis/code/outputmap.png", dpi=200)

map_txt <- apply(mymap, 1:2, function(x) col2rgb(x)[1]/255)
write.table(map_txt, "outputmap.txt", row.names=FALSE, col.names=FALSE)

