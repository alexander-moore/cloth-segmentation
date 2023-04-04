# Twitter bot?
import tweepy

# Your Twitter API keys and access tokens
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create the API object
api = tweepy.API(auth)

def respond_to_mentions():
    # Get the latest mentions of your bot
    mentions = api.mentions_timeline()


    # Loop through each mention and respond with "Hello World!"
    for mention in mentions:
    	mention_text = mention.text

    	media_urls = []
    	if 'media' in mention.entities:
    		for media in mention.entities['media']:
    			media_urls.append(media['media_url'])

        if 'hello' in mention.text.lower():
            reply_text = f'Hello @{mention.user.screen_name}!'
            api.update_status(status=reply_text, in_reply_to_status_id=mention.id) # this looks like how you reply / @ someone


# Loop indefinitely
while True:
    respond_to_mentions()
    time.sleep(60 * 5)  # Check for new mentions every 5 minutes