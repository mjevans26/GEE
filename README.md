<link rel="stylesheet" href="https://cdn.knightlab.com/libs/juxtapose/latest/css/juxtapose.css">

# Google Earth Engine for Conservation

This repo contains example analyses conducted by Defenders of Wildlife using the Google Earth Engine platform.

## Automated Change Detection

We have developed several publicly available apps using the Google Earth Engine App Developer focusing on automatically mapping and quantifying changes to the landscape over time.  The first of these assesses deforestation due to [Logging over time on Prince of Wales Island in the Tongass National Forest](https://defendersofwildlifegis.users.earthengine.app/view/powdeforestation)

<div>
  <img src = "/michael.jpg"/>
</div>

Another example measured coastal habitat loss for several Threatened and Endangered species following [Hurricane Michael](https://defendersofwildlifegis.users.earthengine.app/view/hurricanemichael)

## Landscape Classification
### Ground mounted solar panels in North Carolina

The Nature Conservancy is interested in modelling the impact of the proliferation of ground-mounted solar energy development on wildlife habitat and connectivity in North Carolina.  

We used Sentinel-2 multispectral imagery and Sentinel-1 SAR data to classify the state into Forest, Ag/Pasture, Impervious, Water, and Solar.  Using a random forest classifier, we achieved a Kappa accuracy of 0.84.


<div id="NC-slider" class = "juxtapose" data-startingposition = "30%">
  <img src = "/nlcd.jpg" data-label = "Classification"/>
  <img src = "/S2.jpg" data-label = "Sentinel-2"/>
</div>
<script src="https://cdn.knightlab.com/libs/juxtapose/latest/js/juxtapose.min.js"></script>

### Parking lots on Long Island

Long Island is exploring options for developing utility scale renewable energy as New York State strives to achieve its 50% renewable energy by 2030 goals. In order to promote renewable energy development and habitat conservation, we were interested in finding low-impact sites for solar development on Long Island - and parking lots present an ideal case. 

In this analysis we again used a random forest classifier to 
