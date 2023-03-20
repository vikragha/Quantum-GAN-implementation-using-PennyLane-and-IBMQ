wget http://cactus.nci.nih.gov/download/nci/NCISMA99.sdz
gzip -dc NCISMA99.sdz | awk '{print $2 " NCI" $1}' | sed "s/\[\([BCNOPSF]\)\]/\1/g" | gzip > nci-250k.smi.gz
