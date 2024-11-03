#!/bin/bash

# Create a directory for downloads if it doesn't exist

cd /Users/arikan/Desktop/Research/ml_cuda/inputs/unedited

# Array of URLs to download
urls=(
    "https://snap.stanford.edu/data/facebook_combined.txt.gz" # Snap dataset
    "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz"
    "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz"
    "https://snap.stanford.edu/data/as-skitter.txt.gz"
    "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz"
    "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz"
    "https://snap.stanford.edu/data/roadNet-CA.txt.gz"
    "https://snap.stanford.edu/data/roadNet-PA.txt.gz"
    "https://snap.stanford.edu/data/roadNet-TX.txt.gz"
    "https://snap.stanford.edu/data/facebook_large.zip"
    "https://snap.stanford.edu/data/git_web_ml.zip"
    "https://snap.stanford.edu/data/deezer_europe.zip"
    "https://snap.stanford.edu/data/lastfm_asia.zip"
    "https://snap.stanford.edu/data/twitch_gamers.zip"
    "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz"
    "https://snap.stanford.edu/data/email-Enron.txt.gz"
    "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"
    "https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz" # Suite sparse
    "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/road_central.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/road_usa.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/asia_osm.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/belgium_osm.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/europe_osm.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/germany_osm.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/great-britain_osm.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/italy_osm.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/luxembourg_osm.tar.gz" # Network Repository
    "https://nrvis.com/download/data/bio/bio-CE-CX.zip" # Biological networks
    "https://nrvis.com/download/data/bio/bio-CE-GN.zip"
    "https://nrvis.com/download/data/bio/bio-CE-GT.zip"
    "https://nrvis.com/download/data/bio/bio-CE-HT.zip"
    "https://nrvis.com/download/data/bio/bio-CE-LC.zip"
    "https://nrvis.com/download/data/bio/bio-CE-PG.zip"
    "https://nrvis.com/download/data/bio/bio-DM-CX.zip"
    "https://nrvis.com/download/data/bio/bio-DM-HT.zip"
    "https://nrvis.com/download/data/bio/bio-DM-LC.zip"
    "https://nrvis.com/download/data/bio/bio-DR-CX.zip"
    "https://nrvis.com/download/data/bio/bio-HS-CX.zip"
    "https://nrvis.com/download/data/bio/bio-HS-HT.zip"
    "https://nrvis.com/download/data/bio/bio-HS-LC.zip"
    "https://nrvis.com/download/data/bio/bio-MUTAG_g1.zip"
    "https://nrvis.com/download/data/bio/bio-SC-CC.zip"
    "https://nrvis.com/download/data/bio/bio-SC-GT.zip"
    "https://nrvis.com/download/data/bio/bio-SC-HT.zip"
    "https://nrvis.com/download/data/bio/bio-SC-LC.zip"
    "https://nrvis.com/download/data/bio/bio-SC-TS.zip" # stopped at bio-WormNet
    "https://nrvis.com/download/data/bn/bn-cat-mixed-species_brain_1.zip" # brain networks
    "https://nrvis.com/download/data/bn/bn-fly-drosophila_medulla_1.zip"
    "https://nrvis.com/download/data/bn/bn-human-BNU_1_0025864_session_1-bg.zip"
    "https://nrvis.com/download/data/bn/bn-human-BNU_1_0025865_session_1-bg.zip" # 2 sessions of human bnu
    "https://nrvis.com/download/data/bn/bn-human-Jung2015_M87100374.zip"
    "https://nrvis.com/download/data/bn/bn-human-Jung2015_M87101049.zip" # 2 human jung
    "https://nrvis.com/download/data/bn/bn-macaque-rhesus_brain_1.zip" 
    "https://nrvis.com/download/data/bn/bn-macaque-rhesus_cerebral-cortex_1.zip" 
    "https://nrvis.com/download/data/bn/bn-mouse_brain_1.zip"
    "https://nrvis.com/download/data/bn/bn-mouse_retina_1.zip"
    "https://nrvis.com/download/data/ca/ca-AstroPh.zip" # collaboration networks
    "https://nrvis.com/download/data/ca/ca-CondMat.zip"
    "https://nrvis.com/download/data/ca/ca-HepPh.zip"
    "https://nrvis.com/download/data/ca/ca-IMDB.zip"
    "https://nrvis.com/download/data/ca/ca-MathSciNet.zip"
    "https://nrvis.com/download/data/ca/ca-cit-HepPh.zip"
    "https://nrvis.com/download/data/ca/ca-cit-HepTh.zip"
    "https://nrvis.com/download/data/ca/ca-citeseer.zip"
    "https://nrvis.com/download/data/ca/ca-coauthors-dblp.zip"
    "https://nrvis.com/download/data/ca/ca-hollywood-2009.zip"
    "https://nrvis.com/download/data/ca/ca-netscience.zip"
    "https://nrvis.com/download/data/econ/econ-psmigr1.zip" # economic networks
    "https://nrvis.com/download/data/econ/econ-orani678.zip"
    "https://nrvis.com/download/data/graph500/graph500-scale18-ef16_adj.zip" # graph 500
    "https://nrvis.com/download/data/graph500/graph500-scale19-ef16_adj.zip"
    "https://nrvis.com/download/data/heter/epinions.zip" # heterogenius networks
    "https://nrvis.com/download/data/heter/flickr.zip"
    "https://nrvis.com/download/data/heter/movielens-10m.zip"
    "https://nrvis.com/download/data/heter/visualize-us.zip"
    "https://nrvis.com/download/data/heter/yahoo-msg.zip"
    "https://nrvis.com/download/data/massive/tech-p2p.zip" # massive network data
    "https://nrvis.com/download/data/massive/web-indochina-2004-all.zip"
    "https://nrvis.com/download/data/massive/web-uk-2002-all.zip"
    "https://nrvis.com/download/data/rec/rec-amazon-ratings.zip" # recommendation networks
    "https://nrvis.com/download/data/rec/rec-dating.zip"
    "https://nrvis.com/download/data/rec/rec-eachmovie.zip"
    "https://nrvis.com/download/data/rec/rec-epinions.zip"
    "https://nrvis.com/download/data/rec/rec-github.zip"
    "https://nrvis.com/download/data/rec/rec-libimseti-dir.zip"
    "https://nrvis.com/download/data/rec/rec-movielens.zip"
    "https://nrvis.com/download/data/rec/rec-yahoo-songs.zip"
    "https://nrvis.com/download/data/sc/sc-nasasrb.zip" # scientific computing
    "https://nrvis.com/download/data/sc/sc-pkustk11.zip"
    "https://nrvis.com/download/data/sc/sc-pkustk13.zip"
    "https://nrvis.com/download/data/sc/sc-pwtk.zip"
    "https://nrvis.com/download/data/sc/sc-rel9.zip"
    "https://nrvis.com/download/data/tech/tech-RL-caida.zip" # technological networks
    "https://nrvis.com/download/data/tech/tech-ip.zip" 
    "https://nrvis.com/download/data/tech/tech-p2p.zip"
    "https://nrvis.com/download/data/web/web-BerkStan.zip" # web graphs
    "https://nrvis.com/download/data/web/web-NotreDame.zip"
    "https://nrvis.com/download/data/web/web-Stanford.zip"
    "https://nrvis.com/download/data/web/web-arabic-2005.zip"
    "https://nrvis.com/download/data/web/web-baidu-baike.zip"
    "https://nrvis.com/download/data/bhoslib/frb100-40.zip" # bhoslib
    # Add more URLs here
)

# Function to download file and show progress
download_file() {
    echo "Downloading: $1"
    wget --no-verbose --show-progress "$1"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded: $1"
    else
        echo "Failed to download: $1"
    fi
    echo "------------------------"
}

# Main loop to process downloads
echo "Starting downloads..."
for url in "${urls[@]}"; do
    download_file "$url"
done

echo "Download process completed!"