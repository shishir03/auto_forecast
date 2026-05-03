# Auto-forecast
Trying to train a small language model to generate weather forecasts from model data. The goal is to generate a simplified version of a NWS forecast discussion.

## Installation steps

 - Install `ollama` if you haven't already
 - Run `discussion_processor.py` to generate the discussion dataset
 - Make sure to pull the manifest if you get an error (`ollama pull llama3.1:8b-instruct-q4_K_M`)
 - Run `model.py` to pull the weather model data and to train the language model

## Progress update (April 23, 2026)

The basic training pipeline is now in place, though it isn't great. Currently, I retrieve discussions using [Iowa State's NWS products archive API](https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?help), and process them using the local LLM Llama 3.1. This is done to convert the more meteorologically detailed forecast discussions into a simpler format that a general audience can understand. On my hardware (CPU only), each discussion takes around five minutes, so I did this test run with only around 37 discussions (took around three hours). For now, I'm only working with discussions specifically for the San Francisco Bay Area.

To train the model, I first extract relevant features from GFS model data. This currently includes 250mb jet stream winds, 500mb geopotential height anomalies, and the MSLP (pressure) field. These are then converted into a text format; for now, I output jet streak (i.e. wind maxima in the jet stream) direction and location, low and high pressure centers, and ridge and trough locations and strength. Eventually, I plan to extend this to more parameters by adding a vision encoder for more gridded fields (e.g. temperatures). These text features and the simplified weather forecasts from earlier are provided as examples to the model.

### Current issues
 - Generating the simplified discussions is itself not an easy task. I moved to a two-pass approach to mitigate hallucination possibility, but it still happens sometimes. More prompt engineering / validation on larger datasets will be needed.
 - The small dataset is definitely a big issue, as the model doesn't seem to capture meteorological reasoning very well. The format is also very inconsistent. The model also hallucinates forecasts for locations throughout the US, even though I only trained on weather forecasts for the Bay Area.

### Examples

#### April 4, 2026

**Auto-forecast (the model)**
```
Today:
* Sunny skies with scattered showers and thunderstorms. Highs in the mid-70s.
* Severe weather risk remains low for the rest of the week.

Tomorrow:
* Sunny skies with isolated showers and thunderstorms. Highs in the mid-80s.
* Severe weather risk remains low for the rest of the week.

Friday:
* Sunny skies with scattered showers and thunderstorms. Highs in the mid-80s.
* Severe weather risk remains low for the rest of the week.

Saturday:
* Sunny skies with scattered showers and thunderstorms. Highs in the mid-80s.
* Severe weather risk remains low for the rest of the week.

Sunday:
* Sunny sk
```

**Actual forecast (simplified)**
```
PATTERN: A high-pressure system continues to dominate the region, leading to unusually warm 
temperatures. As it slowly weakens, a trough and low pressure system are expected to build 
into the area from the west.

IMPACTS: For several days, highs have been record-breaking, but that trend is now slowing 
down. Temperatures will still be warmer than usual, but not as extreme. Overnight lows will 
feel slightly more comfortable compared to previous nights. By mid-week, cooler conditions 
are expected with highs dropping 5-10 degrees. Rain is forecasted to return by mid-week, 
potentially bringing an end to the heat.
```

#### April 9, 2026

**Auto-forecast**
```
* Friday – The low pressure system will track southeastward across the region, leading to 
an increase in precipitation on Saturday.

* Saturday – The low pressure system will move southeastward, bringing a change in 
precipitation from rain to snow.

* Sunday – The precipitation will shift to the northwest, leading to a return to rain on 
Monday.

* Monday – The storm will be on the move, but it will be slow-moving and less severe.

* Tuesday – The storm will turn more severe, with heavy rain and strong winds.

* Wednesday – A gradual transition to more moderate weather will take place.

* Thursday – The storm will begin to weaken, but there will be a chance of brief heavy 
rain and gusty winds.

* Friday – The storm will continue to weaken, leading to a return to normal weather conditions.
```

**Actual forecast**
```
PATTERN: A cold front will move into the area on Saturday afternoon and evening, bringing 
rain showers and thunderstorms with it. Behind this system, a low-pressure area will linger, 
leading to scattered rain showers and thunderstorms. By Sunday, the system will continue 
to produce wraparound rain showers and thunderstorms.

IMPACTS: Rain showers and thunderstorms will move into the area on Saturday morning, 
bringing the risk of lightning, heavy rainfall, strong winds, and small hail. The rain is 
expected to be beneficial due to the dry March conditions. As the system moves through, 
widespread rainfall is expected across the region. On Sunday, wraparound rain showers and 
thunderstorms will continue, with a chance of snow at higher elevations. By Monday, 
temperatures are expected to drop below normal, but will rebound to seasonal norms by Thursday.
```

#### April 18, 2026

**Auto-forecast**
```
Temperature trend: The trend is gradually warmer with an increase in average temperature 
across the country.
Precipitation trend: Precipitation is expected to be above normal across most of the country, 
with a slight increase expected in the western parts of the US.
Wind trend: The winds are expected to be mild with a decrease in the southeastern regions.

### Potential Impact:
- Heavy rain in the western parts of the country.
- Strong winds in the southeastern regions.
```

**Actual forecast**
```
PATTERN: A high-pressure system is dominating the weather, leading to warm temperatures 
and increasing moisture from the west.

IMPACTS: Expect very warm temperatures throughout the weekend, especially inland areas 
where it may reach into the 80s. However, clouds will return to coastal areas on Sunday 
due to an approaching low-pressure system. This may cause slight cooling in these regions, 
particularly in the North Bay and San Francisco Bay Area. Rain showers are expected to 
begin Monday morning in the North Bay, spreading southward across the Bay Area. While 
some areas, especially outside of the Central Coast's coastal ranges, may see little 
rainfall on Monday, thunderstorms with up to 30% chance are possible due to increasing 
instability.
```

### Future improvements
 - Biggest thing is just more data / better hardware. It definitely seems like a lot of the limitations of the current model stem from a very small training dataset, so it'll be interesting to see how this changes with more data.
 - If this doesn't work, try more advanced feature extraction, such as gridded temperature and precipitation fields. Maybe the model also needs to know exactly where it's forecasting for.
 - Continue fine-tuning the prompt for simplifying the discussions. I really want to strike the right balance between informative and easy to understand. Currently I feel like the simplified discussions are a bit too simplistic and don't capture enough discussion of the weather pattern.