from bs4 import BeautifulSoup
import os
import requests
import json
import datetime


class CQRI():

    def __init__(self, twitter_path):

        '''

        :param twitter_path: relative path to the twitter.txt file
        '''

        self.twitter_path = twitter_path

        self.tweet_dict = self.parse()


    def parse(self):

        '''
        Returns dict with
        key : Event ID
        val : ([list of tweets ID], label)
        '''


        with open(self.twitter_path) as myfile:

            tweet_dict = {}

            for line in myfile:

                parts = line.rstrip().split(' ')

                #event id
                eid = parts[0].split(':')[1]

                #label
                label = int(parts[1].split(':')[1])

                #tweets
                tweet_ids = [tid for tid in parts[2:]]

                tweet_dict[eid] = (tweet_ids, label)

                #print('building entry', eid, label, tweet_ids)

            return tweet_dict


    def get_dict(self):

        '''
        :return: dictionnary with   key=event_id
                                    val=([tweet_ids], label)
        '''
        return self.tweet_dict


    def download_tweets(self, output_dir):

        '''
        :param output_dir:

        Saves a json file of a dictionnary of {tweet_id : (tweet datetime in raw string format, tweet text)}
        for each event id
        Saves (None, None) if tweet does not exists
        '''

        progress = 0

        tweets_text = {}

        # Create directory for tweets
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for key, val in self.tweet_dict.items():

            tweet_ids = val[0]
            filename = output_dir + '/'  + key + '.json'
            print('Getting tweets of', key)

            for tweet_id in tweet_ids:

                # Writing all tweets to txt files

                # Tweet URL
                tweet_url = 'https://twitter.com/statuses/'+tweet_id
                #Headers for correct date time format
                headers = {"Accept-Language": "en-US"}

                r = requests.get(tweet_url, headers=headers)

                soup = BeautifulSoup(r.text, 'html.parser')

                tweets = soup.findAll('p', class_='tweet-text')
                metadata = soup.findAll('span', class_='metadata')

                if len(tweets) > 0:

                    #parse date time in US format
                    date = metadata[0].text.strip()

                    #Save text
                    tweets_text[tweet_id] = (date, tweets[0].text)

                else:
                    tweets_text[tweet_id] = (None, None)

            # Convert dictionnary to json
            with open(filename, 'w', encoding='utf-8') as myfile:
                json.dump(tweets_text, fp=myfile, ensure_ascii=False)

            print('Progress: {:d} %'.format(int(progress/len(self.tweet_dict))))
            progress += 1


    def get_tweets(self, json_path):

        '''

        :param self:
        :param json_path: path ot .json file of event id
        :return: dictionnary    key=tweet id
                                val=(date in datetime format, tweet string)
        '''

        with open(json_path, 'r', encoding='utf-8') as myfile:
            dict = json.loads(myfile.read())

        #Conversion of dates to datetime
        for key, val in dict.items():
            if val[0] is not None:

                date_parsed = datetime.datetime.strptime(val[0], '%H:%M %p - %d %b %Y')
                dict[key][0] = date_parsed


        return dict




# EXAMPLES OF USE

dataset = CQRI('../Twitter.txt')

dataset.download_tweets('credbank/tweets')

#dict = dataset.get_tweets('credbank/tweets/E17.json')
#print(dict)