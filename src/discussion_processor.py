import ollama
import time

def simplify_discussion(discussion_text):
    start = time.time()
    extraction_response = ollama.chat(
        model='llama3.1:8b-instruct-q4_K_M',
        messages=[
            {
                'role': 'system',
                'content': """Extract every meteorologically significant claim 
                from the following forecast discussion as a bullet list. 
                Quote directly from the text where possible, and do not add any 
                information not present in the text. Only include the bullet list
                in your response."""
            },
            {
                'role': 'user',
                'content': discussion_text
            }
        ]
    )
    extracted_claims = extraction_response['message']['content']
    print(f"{extracted_claims}\n")

    response = ollama.chat(
        model='llama3.1:8b-instruct-q4_K_M',
        messages=[
            {
                'role': 'system',
                'content': """You are a meteorologist providing a weather forecast 
                for a general audience. Translate the following meteorological claims into 
                plain language for a general audience, providing a single summary for the 
                entire forecast period. Do not add any information beyond what is listed.
                
                Your output must follow this exact format:
                PATTERN: 2-3 sentences describing the large-scale synoptic weather pattern
                IMPACTS: 4-5 sentences describing what this means for local weather
                CONFIDENCE: Low, medium, or high
                """
            },
            {
                'role': 'user',
                'content': f"""Translate these claims:\n\n{extracted_claims}. 
                Only include the simplified text in your response."""
            }
        ]
    )

    end = time.time()
    print(f"Total time: {end - start}")
    return response['message']['content']

print(simplify_discussion(
""".SHORT TERM...
Issued at 935 PM PDT Sat Apr 18 2026
(Tonight through Monday)

Satellite imagery shows high clouds continuing to cover the Bay Area
and Central Coast tonight, which are expected to continue to push
through the region in the overnight period before steadily thinning
and scattering Sunday morning. Low temperatures tonight are expected
to hover in the middle 40s to lower 50s across the region, perhaps a
few degrees warmer than the current forecast if the high level cloud
cover is enough to inhibit radiational cooling and reflect thermal
energy back to the surface.

Today will be a day of temperatures close to the seasonal averages,
with the inland valleys reaching highs in the 70s, the Bays seeing
highs in the middle 60s to lower 70s, and the Pacific coast hovering
around the upper 50s to the lower 60s. A gentle onshore breeze with
a southwesterly component will develop during the afternoon, with
the breezy winds persisting into the night as a cold front
approaches the region.

&&

.LONG TERM...
Issued at 935 PM PDT Sat Apr 18 2026
(Monday night through next Saturday)

The focus of the seven-day outlook continues to be centered around
the cold front coming through the Bay Area and Central Coast for the
early part of the work week. Pre-frontal rain showers are expected
to arrive sometime Monday morning across the North Bay and continue
to spread southward through the day, with the main frontal band
coming through later on Monday into Tuesday morning. Behind the
front, and with the associated upper level low coming through
northern California, the newly arrived cold pool will allow for a
chance of isolated to scattered thunderstorms through Tuesday
afternoon and evening, with probabilities ranging from around 20 to
30 percent across the region. Lingering showers and chances for
isolated thunderstorms (up to 15% probability) continue through
Wednesday, and should move out of the region by Wednesday night.
Through all of this, high temperatures will dip into the lower to
middle 60s in the lower elevations to the 50s across the higher
elevations, and rain totals will range from 0.5-1.5" across the
interior valleys and most of the Bay Area and Monterey Bay regions,
to around 1.5-3" in the coastal ranges and the interior mountains of
the North Bay. This should be mostly beneficial across the region,
but some minor nuisance flooding is possible in urban and poor
drainage areas if heavy rain showers or thunderstorms develop.

As the frontal system passes, temperatures will warm slightly into a
rather dry latter part of the week, back to around the seasonal
averages rather similar to today`s highs. Extended guidance from the
Climate Prediction Center leans towards temperatures and rain
totals above seasonal averages for the last week of April."""
))
