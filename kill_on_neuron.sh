neuron-ls | awk '/[0-9]{5,10}/ {for(i=1;i<=NF;i++) if($i ~ /^[0-9]{5,10}$/) print $i}' | xargs -r kill -9
