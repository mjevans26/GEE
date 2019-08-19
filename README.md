---
customjs:
  - https://cdn.knightlab.com/libs/juxtapose/latest/js/juxtapose.min.js 
---
# Google Earth Engine for Conservation

This repo contains example analyses conducted by Defenders of Wildlife using the Google Earth Engine platform.

## Landscape Classification
# Ground mounted solar panels in North Carolina

The Nature Conservancy is interested in modelling the impact of the proliferation of ground-mounted solar energy development on wildlife habitat and connectivity in North Carolina.  

We used Sentinel-2 multispectral imagery and Sentinel-1 SAR data to classify the state into Forest, Ag/Pasture, Impervious, Water, and Solar.  Using a random forest classifier, we achieved a Kappa accuracy of 0.84.

<div id="NC-slider" class = "juxtapose" data-startingposition = "30%">
  <img src = "/nlcd.jpg" data-label = "Classification"/>
  <img src = "/S2.jpg" data-label = "Sentinel-2"/>
</div>

<link rel="stylesheet" href="https://cdn.knightlab.com/libs/juxtapose/latest/css/juxtapose.css">
