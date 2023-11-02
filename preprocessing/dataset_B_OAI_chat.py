import numpy as np
import pandas as pd
#import tensorflow_hub as hub
#import tensorflow as tf
#import spacy 
import openai
import time


TRAIN_DATASET = '../database/train_B_text_processed_embeddings_oai.csv'
OUTPUT_TRAIN_DATASET = '../database/train_B_text_chat.csv'
TEST_DATASET = '../database/test_B_text_processed_embeddings_oai.csv'
OUTPUT_TEST_DATASET = '../database/test_B_text_chat.csv'

openai.api_key = 'YOUR-API-KEY'

REALS = [
    "Local Municipalities Have Right to Ban Hydrofracking",
    "Alibaba Chooses New York Stock Exchange for US IPO Listing",
    "Japan Machine Orders Fall by Record in Spending Caution Sign",
    "Crunchyroll Now Available on Chromecast!",
    "Who Slipped Investors a Chill Pill?",
    "Microsoft Execs Demonstrate Office for iPad App at Press Conference",
    "Jessica Simpson and Eric Johnson Tie Knot in California",
    "House Ways and Means Subcommittee on Health Hearing",
    "Relay for Life Fundraising Kicks Off with Community Event",
    "New Teenage Mutant Ninja Turtles Movie Trailer and Character Posters",
    "Apple still top OEM in US",
    "Gas Prices Continue to Soar; Where to Find it Cheap in La Jolla",
    "Stamp Auctions at Record Price for Fourth Time",
    "T. Rex Comes to Smithsonian for New Dinosaur Hall Exhibit",
    "Upholding Christ and the Constitution",
    "FCC to Release Draft Net Neutrality Rules in May",
    "Xbox One June Update to Bring External Storage & Real Names",
    "China Entrepreneurs Seek Economic Freedom in US",
    "Google Releases its Latest Transparency Report",
    "Courtney Love Hopes She Found the Malaysia Airlines Plane",
    "Apple Posts Solid Gains On Share Repurchases And iPhone Sales",
    "New Website Makes it Easy to Buy Health Insurance Online",
    "New Preferred ETF Off to Fast Start",
    "Shia LaBeouf Books it For Alcohol Treatment",
    "Look for a Meteor Shower Early Tuesday Morning",
    "Apple Beats Combo? The Cost of Cool",
    "Forbes Media to Sell Majority Stake to Group of International Investors",
    "Fed Saw Investors as Too Complacent on Risk as Exit Plan Evolves",
    "Pioneer Brings Apple CarPlay to the Vehicle You Already Own",
    "Why Procter & Gamble Investors Will Continue to Clean Up",
    "Sarah Jessica Parker Returning to Television In New Series",
    "Euro at Risk as Soft Inflation Data Sets the Stage for ECB Stimulus",
    "Supreme Court Conflicted on Legality of Aereo Online Video Service",
    "A Vault Full of Vintage Cash",
    "Fogging for West Nile in San Mateo Wednesday",
    "Samsung Galaxy Tab S Takes On iPad",
    "Alibaba Underwriters Go From Alpha to Beta in IPO",
    "May is Lyme Awareness Month",
    "Trade Gap Shrinks as U.S. Exports Show Global Pickup",
    "Hugh Jackman Heading Back to Broadway",
    "Vials of Smallpox Found Unsecured in NIH Bethesda Storage Room",
    "SEC Asks Municipal Bond Sellers to Report Disclosure Breach",
    "Google Rating Reiterated by Credit Suisse (GOOG)",
    "Atari ET Game Graveyard Unearthed in New Mexico Desert",
    "Gasoline Prices to Rise as Promise in Iraq Upended",
    "Win an Oculus Prize Pack",
    "Project My Screen Ready for Download Now",
    "A Streaker Crashed the Met Ball",
    "Opioid Prescriptions High for US Soldiers Returning From Deployment",
    "Herbalife (HLF) Unaware of Criminal Probe",
    "Walmart Takes Aim at GameStop with New Buyback Program",
    "ECB Draghi Needs European Nations to Be More Involved",
    "BlackBerry Reinforces Mobile Security Leadership with Acquisition of Secusmart",
    "Viagra Promising for Muscular Dystrophy Patients",
    "Pfizer Defends Astra Deal as CEO Braces for Grilling"
]

FAKES = [
    "NASA Discovers Earth-Like Planet in Habitable Zone",
    "Surge in UFO Sightings Sparks Government Investigation",
    "Massive Diamond Found in South African Mine",
    "Scientists Discover New Species in Amazon Rainforest",
    "100-Year-Old Woman Breaks World Record in Sprinting",
    "Elon Musk Unveils Plans for Hyperloop Network",
    "New Study Links Coffee Consumption to Longer Life",
    "Queen Elizabeth II Celebrates 75 Years on the Throne",
    "Zac Efron Stays Mum on Rumors of Romance with Halston Sage",
    "High-Flying Tech Stocks Drop as Markets Worry About Fed Stimulus",
    "GOG to Add Linux Support with 100 Games Available This Fall",
    "Virtual Reality Therapy Revolutionizes Mental Health Care",
    "Bitcoin Revolutionizes Global Economy",
    "Prince Harry and Meghan Markle Welcome Baby Girl",
    "Miley Cyrus Postpones U.S. Tour Dates Until August",
    "Analysts Predict 500,000 Annual Tesla Car Sales by 2020",
    "Captive Killer Whales Defy Lifespan Expectations",
    "Hollywood Actress Reveals Secret Battle with Rare Disease",
    "AI-Powered Chef Creates Culinary Masterpieces",
    "Citi Leads Financials Lower, Stocks Slip in Midday Trade",
    "American Apparel Implements Rights Plan Amid Ongoing Disputes",
    "Recalls Spark Panic and Quick Fixes",
    "Pop Star Ariana Grande Launches Charitable Foundation",
    "Researchers Find Cure for Common Cold",
    "Breakthrough in Fusion Energy Promises Clean Power for All",
    "Self-Driving Trucks Revolutionize Transportation Industry",
    "Historic Peace Summit Brings Nations Together",
    "NASA Discovers Water on Mars: Is Life Possible?",
    "Global Cyberattack Shuts Down Internet in Multiple Countries",
    "Researchers Develop Remote-Control Contraceptive Device",
    "Twitter Agrees to Acquire Data Firm Gnip",
    "Ancient Pyramid Discovered in Antarctica",
    "Elon Musk Announces Plans for Underwater City",
    "Record-Breaking Heatwave Sweeps Across Europe",
    "Hollywood Icon Clint Eastwood Returns to Directing",
    "Archaeologists Discover Ancient Mayan Treasure",
    "Scientists Discover New Species in Unexplored Ocean Depths",
    "Amazon Launches Delivery Service to the Moon",
    "Tokyo Stocks Close 0.6% Higher on Friday Morning",
    "Hollywood A-Listers Join Forces to Save Endangered Species",
    "Prince Harry and Meghan Markle Announce Royal Tour",
    "Coach Faces Challenges in North American Luxury Market",
    "Prince William and Kate Middleton Expecting Twins"
]

PROMPT = """
Given these sentences with label 0, which indicates real headlines:

{reals}

And these sentences with label 1, which means these are fake headlines:

{fakes}

Classify the following sentences accordingly. Classify it as 0 (real) or 1 (fake) depending on the likelihood of the sentence being a real or fake headline. Base your answer on your knowledge and the samples provided above. Give a thoughtful response. Take a deep breath before answering. Justify your answer. Use the following format 

{{justification}}
"Classification::" {{0/1, do not print anything else!}}

Sentence: {sentence}
"""

def ask(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    except openai.error.RateLimitError as err:
        # Wait 1 minute and retry until success
        print(str(err))
        print("Rate limit error, waiting 30s and retrying")
        time.sleep(30)
        return ask(prompt)
    except Exception as err:
        # Wait 1 minute and retry until success
        print(str(err))
        return {'choices': [{'message': {'content': 'X'}}]}
    return response


def predict(input, excluded_list):
    if input not in excluded_list:
        return 2
    num_retry = 3
    has_ended = False
    print(input)
    removed_from_reals, removed_from_fakes = False, False
    if input in REALS:
        REALS.remove(input)
        removed_from_reals = True
    elif input in FAKES:
        FAKES.remove(input)
        removed_from_fakes = True
    prompt = PROMPT.format(reals='\n'.join(REALS), fakes='\n'.join(FAKES), sentence=input)
    while num_retry > 0 and not has_ended:
        response = ask(prompt)
        assistant_response = response['choices'][0]['message']['content']
        print(assistant_response)
        if '0' in assistant_response[-3:]:
            print("    predicted: 0 (real)")
            print(".......................................................")
            if removed_from_reals:
                REALS.append(input)
            elif removed_from_fakes:
                FAKES.append(input)
            return 0
        elif '1' in assistant_response[-3:]:
            print("    predicted: 1 (fake)")
            print(".......................................................")
            if removed_from_reals:
                REALS.append(input)
            elif removed_from_fakes:
                FAKES.append(input)
            return 1
        else:
            num_retry -= 1
    print("Failed to predict :(")
    print(".......................................................")
    if removed_from_reals:
        REALS.append(input)
    elif removed_from_fakes:
        FAKES.append(input)
    return 2


    #print(".......................................................")
    #print(prompt)
    #print(".......................................................")
    return 0

######################### Train dataset #########################
'''
db = pd.read_csv(TRAIN_DATASET)
db = db[db['has_quote_start'] == 0]
db = db[db['has_dots'] == 0]
db = db[db['has_number'] == 0]
db = db[db['has_comma'] == 0]
db = db[db['has_colon'] == 0]
db = db[db['has_parenthesis'] == 0]
db = db[db['has_hyphen'] == 0]
db = db[db['has_and'] == 0]
db = db[db['has_percentage'] == 0]
#db = db[db['has_only_first_upper'] == 0]
db = db[db['has_any_noun_verb_lower'] == 0]
#db = db[db['length'] >= 6]
db = db[db['length'] <= 12]
titles_train = db['Title'].tolist()

db = pd.read_csv(TRAIN_DATASET)
db['Magic bit'] = db['Title'].apply(lambda x: predict(x, titles_train)).apply(pd.Series)

print(db)
pd.DataFrame.to_csv(db, OUTPUT_TRAIN_DATASET)
'''
######################### Test dataset #########################

db = pd.read_csv(TEST_DATASET)
db = db[db['has_quote_start'] == 0]
db = db[db['has_dots'] == 0]
db = db[db['has_number'] == 0]
db = db[db['has_comma'] == 0]
db = db[db['has_colon'] == 0]
db = db[db['has_parenthesis'] == 0]
db = db[db['has_hyphen'] == 0]
db = db[db['has_and'] == 0]
db = db[db['has_percentage'] == 0]
#db = db[db['has_only_first_upper'] == 0]
db = db[db['has_any_noun_verb_lower'] == 0]
db = db[db['length'] >= 6]
db = db[db['length'] <= 12]
titles_test = db['Title'].tolist()

print("length: ", len(titles_test))

db = pd.read_csv(TEST_DATASET)
db['Magic bit'] = db['Title'].apply(lambda x: predict(x, titles_test)).apply(pd.Series)

print(db)
pd.DataFrame.to_csv(db, OUTPUT_TEST_DATASET)



